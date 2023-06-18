from silero_tts import SileroTTS
from chat import Chat

# Create a new chat session
#chat = Chat(model_path="./models/llm/ggml-v3-13b-hermes-q5_1.bin", identity_path="./identities/chelsea.txt")
chat = Chat(model_path="./models/llm/ggml-v3-13b-hermes-q5_1.bin")

# tts = SileroTTS("ru", "v3_1_ru", "baya")
tts = SileroTTS("en", "v3_en", "en_21")

while True:
    user_message = input('User: ')
    
    if user_message.lower() == 'quit':
        break

    # Send a message to the chatbot
    response = chat.send_message(user_message)
    print(f'ChatBot: {response}')
    tts.generate_and_play_audio(response)
