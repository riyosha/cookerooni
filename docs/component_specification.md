# Software Components 

## Ingredient Identifier: Object Recognition  
Input: User uploaded image with ingredients    
Output: JSON formatted listed of ingredients with their (i) quantity, (ii) nutritional value.   

## Recipe Provider  
Input: (i) JSON formatted listed of ingredients with their quantity, nutritional value.  
(ii) User preferences (like prioritization of recipe time or recipe nutrition) 
Output: 5 recipes for creating dishes with available ingredients, ranked according to user preferences.

## Context Manager 

## Recipe Vector Database


## Data Sources

1. Recipe Data Sources:
- Recipe1M+ (images + recipe data)
- RecipeNLG (https://www.kaggle.com/datasets/paultimothymooney/recipenlg)
  - Builds on Recipe1M+ data. We don't need image embeddings in our vector DB, so this seems ideal.
