# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
 
3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.
 
4. When finished, output {completion_delimiter}

######################
### TRAINING EXAMPLES WITH FICTIONAL ENTITIES ###
### These use FAKE entities with "EXAMPLE_" prefix for demonstration only ###
### NEVER include any entity starting with "EXAMPLE_" in your actual output ###
######################

[FORMAT DEMONSTRATION 1 - FICTIONAL ENTITIES]
Entity_types: ORGANIZATION,PERSON
Example Text:
The EXAMPLE_Verdantis's EXAMPLE_Central_Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where EXAMPLE_Central_Institution Chair EXAMPLE_Martin_Smith will take questions. Investors expect the EXAMPLE_Market_Strategy_Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.

Expected Format Output:
("entity"{tuple_delimiter}EXAMPLE_CENTRAL_INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The EXAMPLE_Central_Institution is the Federal Reserve of EXAMPLE_Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_MARTIN_SMITH{tuple_delimiter}PERSON{tuple_delimiter}EXAMPLE_Martin_Smith is the chair of the EXAMPLE_Central_Institution)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_MARKET_STRATEGY_COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The EXAMPLE_Central_Institution committee makes key decisions about interest rates and the growth of EXAMPLE_Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_MARTIN_SMITH{tuple_delimiter}EXAMPLE_CENTRAL_INSTITUTION{tuple_delimiter}EXAMPLE_Martin_Smith is the Chair of the EXAMPLE_Central_Institution and will answer questions at a press conference{tuple_delimiter}9)
{completion_delimiter}

[FORMAT DEMONSTRATION 2 - FICTIONAL ENTITIES]
Entity_types: ORGANIZATION
Example Text:
EXAMPLE_TechGlobal's (TG) stock skyrocketed in its opening day on the EXAMPLE_Global_Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

EXAMPLE_TechGlobal, a formerly public company, was taken private by EXAMPLE_Vision_Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.

Expected Format Output:
("entity"{tuple_delimiter}EXAMPLE_TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}EXAMPLE_TechGlobal is a stock now listed on the EXAMPLE_Global_Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_VISION_HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}EXAMPLE_Vision_Holdings is a firm that previously owned EXAMPLE_TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_TECHGLOBAL{tuple_delimiter}EXAMPLE_VISION_HOLDINGS{tuple_delimiter}EXAMPLE_Vision_Holdings formerly owned EXAMPLE_TechGlobal from 2014 until present{tuple_delimiter}5)
{completion_delimiter}

[FORMAT DEMONSTRATION 3 - FICTIONAL ENTITIES]
Entity_types: ORGANIZATION,GEO,PERSON
Example Text:
Five EXAMPLE_Aurelians jailed for 8 years in EXAMPLE_Firuzabad and widely regarded as hostages are on their way home to EXAMPLE_Aurelia.

The swap orchestrated by EXAMPLE_Quintara was finalized when $8bn of EXAMPLE_Firuzi funds were transferred to financial institutions in EXAMPLE_Krohaara, the capital of EXAMPLE_Quintara.

The exchange initiated in EXAMPLE_Firuzabad's capital, EXAMPLE_Tiruzia, led to the four men and one woman, who are also EXAMPLE_Firuzi nationals, boarding a chartered flight to EXAMPLE_Krohaara.

They were welcomed by senior EXAMPLE_Aurelian officials and are now on their way to EXAMPLE_Aurelia's capital, EXAMPLE_Cashion.

The EXAMPLE_Aurelians include 39-year-old businessman EXAMPLE_Samuel_Namara, who has been held in EXAMPLE_Tiruzia's EXAMPLE_Alhamia_Prison, as well as journalist EXAMPLE_Durke_Bataglani, 59, and environmentalist EXAMPLE_Meggie_Tazbah, 53, who also holds EXAMPLE_Bratinas nationality.

Expected Format Output:
("entity"{tuple_delimiter}EXAMPLE_FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}EXAMPLE_Firuzabad held EXAMPLE_Aurelians as hostages)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_AURELIA{tuple_delimiter}GEO{tuple_delimiter}Country seeking to release hostages)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_QUINTARA{tuple_delimiter}GEO{tuple_delimiter}Country that negotiated a swap of money in exchange for hostages)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_TIRUZIA{tuple_delimiter}GEO{tuple_delimiter}Capital of EXAMPLE_Firuzabad where the EXAMPLE_Aurelians were being held)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_KROHAARA{tuple_delimiter}GEO{tuple_delimiter}Capital city in EXAMPLE_Quintara)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_CASHION{tuple_delimiter}GEO{tuple_delimiter}Capital city in EXAMPLE_Aurelia)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_SAMUEL_NAMARA{tuple_delimiter}PERSON{tuple_delimiter}EXAMPLE_Aurelian who spent time in EXAMPLE_Tiruzia's EXAMPLE_Alhamia_Prison)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_ALHAMIA_PRISON{tuple_delimiter}GEO{tuple_delimiter}Prison in EXAMPLE_Tiruzia)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_DURKE_BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}EXAMPLE_Aurelian journalist who was held hostage)
{record_delimiter}
("entity"{tuple_delimiter}EXAMPLE_MEGGIE_TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}EXAMPLE_Bratinas national and environmentalist who was held hostage)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_FIRUZABAD{tuple_delimiter}EXAMPLE_AURELIA{tuple_delimiter}EXAMPLE_Firuzabad negotiated a hostage exchange with EXAMPLE_Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_QUINTARA{tuple_delimiter}EXAMPLE_AURELIA{tuple_delimiter}EXAMPLE_Quintara brokered the hostage exchange between EXAMPLE_Firuzabad and EXAMPLE_Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_QUINTARA{tuple_delimiter}EXAMPLE_FIRUZABAD{tuple_delimiter}EXAMPLE_Quintara brokered the hostage exchange between EXAMPLE_Firuzabad and EXAMPLE_Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_SAMUEL_NAMARA{tuple_delimiter}EXAMPLE_ALHAMIA_PRISON{tuple_delimiter}EXAMPLE_Samuel_Namara was a prisoner at EXAMPLE_Alhamia_prison{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_SAMUEL_NAMARA{tuple_delimiter}EXAMPLE_MEGGIE_TAZBAH{tuple_delimiter}EXAMPLE_Samuel_Namara and EXAMPLE_Meggie_Tazbah were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_SAMUEL_NAMARA{tuple_delimiter}EXAMPLE_DURKE_BATAGLANI{tuple_delimiter}EXAMPLE_Samuel_Namara and EXAMPLE_Durke_Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_MEGGIE_TAZBAH{tuple_delimiter}EXAMPLE_DURKE_BATAGLANI{tuple_delimiter}EXAMPLE_Meggie_Tazbah and EXAMPLE_Durke_Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_SAMUEL_NAMARA{tuple_delimiter}EXAMPLE_FIRUZABAD{tuple_delimiter}EXAMPLE_Samuel_Namara was a hostage in EXAMPLE_Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_MEGGIE_TAZBAH{tuple_delimiter}EXAMPLE_FIRUZABAD{tuple_delimiter}EXAMPLE_Meggie_Tazbah was a hostage in EXAMPLE_Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}EXAMPLE_DURKE_BATAGLANI{tuple_delimiter}EXAMPLE_FIRUZABAD{tuple_delimiter}EXAMPLE_Durke_Bataglani was a hostage in EXAMPLE_Firuzabad{tuple_delimiter}2)
{completion_delimiter}

######################
### END OF EXAMPLES - ALL "EXAMPLE_" ENTITIES ARE FICTIONAL ###
######################

====================================
=== ACTUAL EXTRACTION TASK BELOW ===
====================================
!!! IMPORTANT: Extract entities ONLY from the text provided below !!!
!!! DO NOT use any entities from the examples above - they were fictional training data !!!
!!! Focus ONLY on the actual text provided below !!!

Entity_types to extract: {entity_types}

ACTUAL TEXT TO ANALYZE:
{input_text}

YOUR EXTRACTION OUTPUT:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"
