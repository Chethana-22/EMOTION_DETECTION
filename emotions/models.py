from datetime import datetime



class Prediction:
    def __init__(self,user_input_type,user_input,user):
        self.user_input_type = user_input_type
        self.user_input = user_input
        self.user = user

class Remedy:
    def __init__(self,remedy_type,remedy_priority,remedy_link):
        self.remedy_type = remedy_type
        self.remedy_priority = remedy_priority
        self.remedy_link = remedy_link

