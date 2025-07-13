import pandas as pd

class SignUp:
    def __init__(self, username, password, survey_data):
        self.username = username
        self.password = password
        self.survey_data = survey_data
        self.df = pd.read_csv('data/users.csv')
        self.df['username'] = username
        self.df['password'] = password

    def add_survey_data(self):
        """Expected columns:
        'Age', 'Gender', 'Diagnosis', 'Symptom Severity (1-10)', 'Mood Score (1-10)', 'Sleep Quality (1-10)', 'Physical Activity (hrs/week)' , 
        'Medication', 'Therapy Type', 'Treatment Start Date', 'Treatment Duration (weeks)', 'Stress Level (1-10)', 'Outcome', 
        'Treatment Progress (1-10)', 'AI-Detected Emotional State', 'Adherence to Treatment (%)']
        """
        for column_name, item in self.survey_data.items():
            if column_name not in self.df.columns:
                raise ValueError(f"Column '{column_name}' not found in the user data.")
            self.df.at[0, column_name] = item
    
    def save(self):
        self.df.to_csv('data/users.csv', index=False)
        print(f"User '{self.username}' data saved successfully.")

class Login:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.df = pd.read_csv('data/users.csv')

    def authenticate(self):
        user_data = self.df[(self.df['username'] == self.username) & (self.df['password'] == self.password)]
        if not user_data.empty:
            print(f"User '{self.username}' logged in successfully.")
            return True
        else:
            print("Invalid username or password.")
            return False
        
    def get_user_data(self):
        user_data = self.df[self.df['username'] == self.username]
        if not user_data.empty:
            return user_data.iloc[0].to_dict()
        else:
            print("User data not found.")
            return None
        
