# commands

- curl.exe -X POST -F "file=@testfile5.pdf" http://localhost:8000/api/v1/analyze
- curl.exe http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/status
- curl.exe http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result
- (Invoke-WebRequest http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result/markdown).Content
- (Invoke-WebRequest http://localhost:8000/api/v1/jobs/5bdd1fd9-2c7c-4baf-b39a-f18bc5205fb0/result/markdown).Content | Out-File result1.md
