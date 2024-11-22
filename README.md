
#  Pharmabot: A natural response chatbot for drugclassification and information 🤖.

 Created a medical chatbot for classify scheduled non-scheduled and new drug.
![google_palm2_hero_1](https://github.com/code-red-Marshall/Pharmabot--LLM-chatbot/assets/82904501/6f377da5-54ec-444e-982d-3136664b49c9)




## Files in the Repository
- `main.py`: The main Python script that executes the core functionalities of this project.

- `data.csv`: The final data that is converted to embeddings using huggingface embeddings (model_name='sentence-transformers/all-MiniLM-L6-v2').
- `Streamlit`: It contain the code for the chatbot deployment.it make accesible and user-friendly for a simple web interface. 
- `pharmatrack_non-scheduled.csv`: Database with data related to non-scheduled pharmaceutical drugs and their tracking information.

## Business Problem: 
Developing a medical chatbot to categorize medicines  like scheduled,non-scheduled and new drug classifications.
## Business Success Criteria :
Increase customer engagement on the website by 30%
## Economic Success Criteria :
Increased customer engagement on the website, contributing to an approximate annual revenue growth of $20,000 in the healthcare sector.

## Chatbot Architecture: 

![image](https://github.com/user-attachments/assets/d0c198b5-968c-4949-af21-1012e7d6b9c1)

## Bot Flow:

Step 1 : The user rise the queries related to the drugs.

Step 2 : The bot checks the information related drung if found in dataset then answer based to the query.

Step 3: If the drug is not found in the dataset, it will be considered as a new drug directly. And the bot asks the user to contact the team for further pricing details.
 
## Setup

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install all dependencies listed in `requirements.txt` using `pip install -r requirements.txt`.
4. Run the main.py to initiate the bot and start asking queries for the formulations present in the csv file named "data".
