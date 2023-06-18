import os
from llama_cpp import Llama
from typing import Optional

class Chat:
    def __init__(self, model_path: str, identity_path: str = None):
        self.chat_log = ''
        self.identity = None
        self.n_ctx = 512
        self.max_tokens = 32
        self.n_threads = 8
        self.llm = Llama(model_path=model_path, n_ctx = self.n_ctx, n_threads=self.n_threads, verbose=False)
        
        if identity_path is not None:
            self.set_identity(identity_path)
        
    def set_identity(self, identity_path):
        # Load the identity from the text file and use it as the first message
        if identity_path and os.path.isfile(identity_path):
            with open(identity_path, 'r') as file:
                identity = file.read()
                self.identity = identity
                self.chat_log = f'Bot: {self.identity}\n'

    def create_prompt(self, message: str) -> str:
        self.chat_log += f'User: {message}\n'
        prompt = self.chat_log + 'Bot: '

        # Tokenize the prompt
        prompt_tokens = self.llm.tokenize(prompt.encode())

        # !! TO DO !! I'm currently removing what was said and keeping only the personality of the assistant
        # would be great to find a more efficient way to do it while preserving the context of the conversation
        # If the number of tokens in the prompt exceeds n_ctx
        size = len(prompt_tokens) + self.max_tokens

        if size >= self.n_ctx:
            # If an identity was defined reset to identity
            if self.identity is not None:
                self.chat_log = f'Bot: {self.identity}\n'
            else:
                self.chat_log = ''

        self.chat_log += f'User: {message}\n'
        prompt = self.chat_log + 'Bot: '
        
        return prompt
            
    def send_message(self, message: str) -> str:
        prompt = self.create_prompt(message)
        output = self.llm(prompt, max_tokens=self.max_tokens, echo=True)
        response = output['choices'][0]['text'].split('Bot: ')[-1]
        self.chat_log += f'Bot: {response}\n'
        return response

    def get_chat_log(self) -> str:
        return self.chat_log
