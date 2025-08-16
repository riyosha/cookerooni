# Software Components 

## Ingredient Identifier
We have two options here:
Easier: Image Classification Model
**Input**: User uploaded images of each separate ingredient    
**Output**: JSON formatted listed of ingredients classified with their nutritional value.   

Trickier: Object Detection Model
**Input**: 1 or few user uploaded images with multiple ingredients in an image.
**Output**: JSON formatted listed of ingredients with their (i) quantity (hard), (ii) nutritional value.   



## Recipe Provider  
**Input**: (i) JSON formatted listed of ingredients with their quantity, nutritional value.  
(ii) User preferences (like prioritization of recipe time or recipe nutrition)   
**Output**: 5 recipes for creating dishes with available ingredients, ranked according to user preferences.

## Context Manager 

## Recipe Vector Database
**Input**: Dataframe with textual recipe data  
**Output**: A vector DB of recipes that can be used for RAG searches.

## User Interface 
**Input**: User preferences, user images  
**Output**: Top 5 Recipes (each recipe has its ingredients, instructions, nutritional value, and recipe link attached)


## Data Sources

1. Recipe Data Sources:
- Recipe1M+ (images + recipe data)
- RecipeNLG (https://www.kaggle.com/datasets/paultimothymooney/recipenlg)
  - Builds on Recipe1M+ data. We don't need image embeddings in our vector DB, so this seems ideal.
 
2. Ingredient image Data Scources:
- 
