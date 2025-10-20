# Exact Code Changes Required - Stale Task Pipeline

This document shows the EXACT changes needed in unified diff format.

## File: /home/tuomo/code/fileintel/src/fileintel/celery_config.py

### Change 1: Add logger initialization (after line 11)

```diff
 from celery import Celery
 from kombu import Queue
 from fileintel.core.config import get_config
 import logging
+
+logger = logging.getLogger(__name__)

 # Create Celery application instance
 app = Celery("fileintel")
```

---

### Change 2: Update datetime import (line 358)

```diff
 from celery.signals import (
     task_success,
     task_failure,
     task_retry,
     worker_ready,
     celeryd_after_setup,
     task_prerun,
     task_postrun,
 )
-from datetime import datetime, timedelta
+from datetime import datetime, timedelta, timezone
```

---

### Change 3: Add serialization helper (before line 368)

```diff
 from datetime import datetime, timedelta, timezone

+
+def _serialize_task_args(data):
+    """
+    Safely serialize task arguments for JSONB storage.
+
+    Handles non-JSON-serializable types (sets, custom objects, etc.)
+    by converting them to string representation.
+    """
+    import json
+    try:
+        # Test if directly serializable to JSON
+        json.dumps(data)
+        return data
+    except (TypeError, ValueError):
+        # Convert to string representation for non-serializable types
+        try:
+            return str(data)[:1000]  # Limit size to prevent database bloat
+        except Exception:
+            return "<unserializable>"
+

 def _get_task_registry_session():
     """Get a database session for task registry operations."""
```

---

### Change 4: Fix task_started_handler (lines 375-408)

```diff
 @task_prerun.connect
 def task_started_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
     """Track task start in database registry."""
     try:
         from fileintel.storage.models import CeleryTaskRegistry
         import os

         session = _get_task_registry_session()
         try:
             # Get worker info
             worker_id = task.request.hostname if task and hasattr(task, 'request') else 'unknown'
             worker_pid = os.getpid()

             # Create or update task registry entry
             task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()

             if task_entry:
                 # Update existing entry
                 task_entry.status = 'STARTED'
-                task_entry.started_at = datetime.utcnow()
+                task_entry.started_at = datetime.now(timezone.utc)
                 task_entry.worker_id = worker_id
                 task_entry.worker_pid = worker_pid
-                task_entry.last_heartbeat = datetime.utcnow()
+                task_entry.last_heartbeat = datetime.now(timezone.utc)
             else:
                 # Create new entry
                 task_entry = CeleryTaskRegistry(
                     task_id=task_id,
                     task_name=sender.name if sender else 'unknown',
                     worker_id=worker_id,
                     worker_pid=worker_pid,
                     status='STARTED',
-                    started_at=datetime.utcnow(),
-                    last_heartbeat=datetime.utcnow(),
-                    args=args,
-                    kwargs=kwargs,
+                    started_at=datetime.now(timezone.utc),
+                    last_heartbeat=datetime.now(timezone.utc),
+                    args=_serialize_task_args(args),
+                    kwargs=_serialize_task_args(kwargs),
                 )
                 session.add(task_entry)

             session.commit()
+        except Exception:
+            session.rollback()
+            raise
         finally:
             session.close()

     except Exception as e:
         logger.error(f"Error tracking task start for {task_id}: {e}")
         # Don't fail the task due to tracking issues
```

---

### Change 5: Fix task_success_handler (lines 423-432)

```diff
 @task_success.connect
 def task_success_handler(
     sender=None, task_id=None, result=None, retries=None, einfo=None, **kwargs
 ):
     """Handle successful task completion and update registry."""
     try:
         from fileintel.storage.models import CeleryTaskRegistry

         session = _get_task_registry_session()
         try:
             task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()
             if task_entry:
                 task_entry.status = 'SUCCESS'
-                task_entry.completed_at = datetime.utcnow()
-                task_entry.result = {'success': True, 'result': str(result)[:1000]}  # Limit size
+                task_entry.completed_at = datetime.now(timezone.utc)
+                task_entry.result = _serialize_task_args(
+                    {'success': True, 'result': str(result)[:1000]}
+                )
                 session.commit()
+        except Exception:
+            session.rollback()
+            raise
         finally:
             session.close()

     except Exception as e:
         logger.error(f"Error updating task success for {task_id}: {e}")
```

---

### Change 6: Fix task_failure_handler (lines 446-455)

```diff
 @task_failure.connect
 def task_failure_handler(
     sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs
 ):
     """Handle task failures and update registry."""
     try:
         from fileintel.storage.models import CeleryTaskRegistry

         session = _get_task_registry_session()
         try:
             task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()
             if task_entry:
                 task_entry.status = 'FAILURE'
-                task_entry.completed_at = datetime.utcnow()
-                task_entry.result = {'error': str(exception)[:1000]}  # Limit size
+                task_entry.completed_at = datetime.now(timezone.utc)
+                task_entry.result = _serialize_task_args(
+                    {'error': str(exception)[:1000]}
+                )
                 session.commit()
+        except Exception:
+            session.rollback()
+            raise
         finally:
             session.close()

     except Exception as e:
         logger.error(f"Error updating task failure for {task_id}: {e}")
```

---

### Change 7: Fix task_retry_handler (lines 467-475)

```diff
 @task_retry.connect
 def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwargs):
     """Handle task retries and update registry."""
     try:
         from fileintel.storage.models import CeleryTaskRegistry

         session = _get_task_registry_session()
         try:
             task_entry = session.query(CeleryTaskRegistry).filter_by(task_id=task_id).first()
             if task_entry:
                 task_entry.status = 'RETRY'
-                task_entry.last_heartbeat = datetime.utcnow()
+                task_entry.last_heartbeat = datetime.now(timezone.utc)
                 session.commit()
+        except Exception:
+            session.rollback()
+            raise
         finally:
             session.close()

     except Exception as e:
         logger.error(f"Error updating task retry for {task_id}: {e}")
```

---

### Change 8: Fix cleanup_stale_tasks - Part A (lines 481-505)

```diff
 @worker_ready.connect
 def cleanup_stale_tasks(sender=None, **kwargs):
     """
     Clean up stale tasks on worker startup.

     When workers are forcibly terminated (e.g., docker-compose down),
     tasks can be left in STARTED state in the database. This handler:
     1. Finds tasks in STARTED state from workers that are no longer alive
     2. Revokes those tasks so they can be retried or cleaned up
     """
     logger.info("Worker ready - checking for stale tasks in database...")

     try:
         from fileintel.storage.models import CeleryTaskRegistry

         # Get currently active workers
         inspect = app.control.inspect()
         stats = inspect.stats()

-        if stats:
-            active_worker_ids = set(stats.keys())
-            logger.info(f"Active workers: {active_worker_ids}")
-        else:
-            active_worker_ids = set()
-            logger.warning("No active workers found via inspect - may be a timing issue")
+        if not stats:
+            logger.warning(
+                "No active workers found via inspect - skipping cleanup to avoid "
+                "revoking tasks from workers that are still starting up"
+            )
+            return
+
+        active_worker_ids = set(stats.keys())
+        logger.info(f"Active workers: {active_worker_ids}")
```

---

### Change 9: Fix cleanup_stale_tasks - Part B (lines 508-565)

```diff
         # Query database for tasks in STARTED or RETRY state
         session = _get_task_registry_session()
         try:
             stale_tasks = (
                 session.query(CeleryTaskRegistry)
                 .filter(CeleryTaskRegistry.status.in_(['STARTED', 'RETRY']))
                 .all()
             )

             if not stale_tasks:
                 logger.info("No tasks in STARTED/RETRY state found")
                 return

+            # Safety: Minimum time before considering task stale
+            MINIMUM_STALE_TIME = timedelta(minutes=5)
+
             stale_count = 0
             for task_entry in stale_tasks:
+                # Skip tasks that started very recently (might be from starting workers)
+                if task_entry.started_at:
+                    running_time = datetime.now(timezone.utc) - task_entry.started_at
+                    if running_time < MINIMUM_STALE_TIME:
+                        continue
+
                 # Check if worker is still alive
                 if task_entry.worker_id not in active_worker_ids:
                     logger.warning(
                         f"Found stale task {task_entry.task_id} from dead worker "
                         f"{task_entry.worker_id} - revoking"
                     )

                     # Revoke the task
                     try:
                         app.control.revoke(task_entry.task_id, terminate=False)

                         # Mark as revoked in database
                         task_entry.status = 'REVOKED'
-                        task_entry.completed_at = datetime.utcnow()
-                        task_entry.result = {
+                        task_entry.completed_at = datetime.now(timezone.utc)
+                        task_entry.result = _serialize_task_args({
                             'error': f'Worker {task_entry.worker_id} died unexpectedly'
-                        }
+                        })
                         session.commit()

                         stale_count += 1

                     except Exception as revoke_error:
                         logger.error(
                             f"Error revoking stale task {task_entry.task_id}: {revoke_error}"
                         )
                 else:
                     # Worker is alive - check heartbeat for very old tasks
                     if task_entry.last_heartbeat:
-                        time_since_heartbeat = datetime.utcnow() - task_entry.last_heartbeat
+                        time_since_heartbeat = datetime.now(timezone.utc) - task_entry.last_heartbeat
                         # If no heartbeat for 6 hours, consider it stale
                         if time_since_heartbeat > timedelta(hours=6):
                             logger.warning(
                                 f"Task {task_entry.task_id} has no heartbeat for "
                                 f"{time_since_heartbeat} - may be stuck"
                             )
                             # Don't auto-revoke - just log warning for manual investigation

             if stale_count > 0:
                 logger.info(f"Revoked {stale_count} stale tasks from dead workers")
             else:
                 logger.info("No stale tasks found requiring cleanup")

+        except Exception:
+            session.rollback()
+            raise
         finally:
             session.close()

     except Exception as e:
         logger.error(f"Error during stale task cleanup: {e}")
         # Don't fail worker startup due to cleanup issues

     logger.info("Stale task check complete")
```

---

## File: /home/tuomo/code/fileintel/src/fileintel/cli/tasks.py

### Change 10: Apply same fixes to CLI cleanup command (lines 238-340)

```diff
 @app.command("cleanup-stale")
 def cleanup_stale_tasks(
     dry_run: bool = typer.Option(
         True, "--dry-run/--execute", help="Show what would be cleaned without doing it."
     ),
     max_age_hours: int = typer.Option(
         6, "--max-age-hours", help="Consider tasks stale if no heartbeat for this many hours."
     ),
 ):
     """
     Clean up stale tasks from dead workers.

     When workers die unexpectedly (docker-compose down, crashes), tasks can be
     left in STARTED state. This command finds and revokes such tasks.
     """
     from fileintel.storage.models import CeleryTaskRegistry, SessionLocal
     from celery import current_app
-    from datetime import datetime, timedelta
+    from datetime import datetime, timedelta, timezone

     cli_handler.console.print("[bold blue]Scanning for stale tasks...[/bold blue]")

     try:
         # Get active workers
         inspect = current_app.control.inspect()
         stats = inspect.stats()

-        if stats:
-            active_worker_ids = set(stats.keys())
-            cli_handler.console.print(f"Active workers: {', '.join(active_worker_ids)}")
-        else:
-            active_worker_ids = set()
-            cli_handler.console.print("[yellow]Warning: No active workers found[/yellow]")
+        if not stats:
+            cli_handler.console.print(
+                "[yellow]Warning: No active workers found - aborting to avoid "
+                "revoking tasks from workers that may be starting[/yellow]"
+            )
+            return
+
+        active_worker_ids = set(stats.keys())
+        cli_handler.console.print(f"Active workers: {', '.join(active_worker_ids)}")

         # Query database
         session = SessionLocal()
         try:
             stale_tasks = (
                 session.query(CeleryTaskRegistry)
                 .filter(CeleryTaskRegistry.status.in_(['STARTED', 'RETRY']))
                 .all()
             )

             if not stale_tasks:
                 cli_handler.console.print("[green]No tasks in STARTED/RETRY state[/green]")
                 return

             cli_handler.console.print(f"Found {len(stale_tasks)} tasks to check")

+            MINIMUM_STALE_TIME = timedelta(minutes=5)
             stale_count = 0
             heartbeat_warnings = 0

             for task_entry in stale_tasks:
                 is_stale = False
                 reason = ""
+
+                # Skip very recent tasks
+                if task_entry.started_at:
+                    running_time = datetime.now(timezone.utc) - task_entry.started_at
+                    if running_time < MINIMUM_STALE_TIME:
+                        continue

                 # Check if worker is dead
                 if task_entry.worker_id not in active_worker_ids:
                     is_stale = True
                     reason = f"Worker {task_entry.worker_id} is dead"

                 # Check heartbeat age
                 elif task_entry.last_heartbeat:
-                    age = datetime.utcnow() - task_entry.last_heartbeat
+                    age = datetime.now(timezone.utc) - task_entry.last_heartbeat
                     if age > timedelta(hours=max_age_hours):
                         is_stale = True
                         reason = f"No heartbeat for {age}"
                         heartbeat_warnings += 1

                 if is_stale:
                     stale_count += 1
                     cli_handler.console.print(
                         f"  [yellow]Stale:[/yellow] {task_entry.task_id[:12]}... "
                         f"({task_entry.task_name}) - {reason}"
                     )

                     if not dry_run:
                         # Revoke the task
                         current_app.control.revoke(task_entry.task_id, terminate=False)

                         # Update database
                         task_entry.status = 'REVOKED'
-                        task_entry.completed_at = datetime.utcnow()
+                        task_entry.completed_at = datetime.now(timezone.utc)
                         task_entry.result = {'error': f'Cleaned up: {reason}'}
                         session.commit()

                         cli_handler.console.print(f"    [green]Revoked[/green]")

             if dry_run:
                 cli_handler.console.print(
                     f"\n[bold yellow]DRY RUN:[/bold yellow] Found {stale_count} stale tasks"
                 )
                 cli_handler.console.print("Use --execute to actually revoke them")
             else:
                 cli_handler.console.print(
                     f"\n[bold green]Revoked {stale_count} stale tasks[/bold green]"
                 )

+        except Exception:
+            session.rollback()
+            raise
         finally:
             session.close()

     except Exception as e:
         cli_handler.display_error(f"Error during cleanup: {e}")
         raise typer.Exit(1)
```

---

## Summary of Changes

**Total files modified:** 2
- `/home/tuomo/code/fileintel/src/fileintel/celery_config.py`
- `/home/tuomo/code/fileintel/src/fileintel/cli/tasks.py`

**Total changes:** 10 distinct modifications
- 1 logger initialization
- 1 import update
- 1 helper function added
- 7 fixes to handlers and cleanup logic

**Lines of code added:** ~50
**Lines of code modified:** ~30
**Estimated time:** 90 minutes including testing
