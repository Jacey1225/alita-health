import pandas as pd

class UpdateJournal:
    def __init__(self, username, entry, filename="data/journals.csv"):
        self.username = username
        self.entry = entry
        self.filename = filename
        self.journal_df = pd.read_csv(self.filename) 
        if self.username not in self.journal_df['username'].values:
            self.journal_df['username'] = self.username

    def add_entry(self):
        new_entry = pd.DataFrame({
            'username': [self.username],
            'entry': [self.entry]
        })
        self.journal_df = pd.concat([self.journal_df, new_entry], ignore_index=True)

    def save(self):
        self.journal_df.to_csv(self.filename, index=False)