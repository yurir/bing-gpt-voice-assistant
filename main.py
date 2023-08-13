# coding: utf8
import asyncio
# remove whisper warning
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import json
import logging
import os
import re
import time

import boto3
import openai
import pydub
import speech_recognition as sr
import whisper
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from pydub import playback
import webbrowser
import requests
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# import vlc
# import pafy

log = logging.getLogger("app")
log.setLevel(logging.DEBUG)
# log.addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# use gTTS to convert text to speech
from gtts import gTTS

# import os

# Initialize the OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()

BING_WAKE_WORD = json.loads(os.environ["BING_WAKE_WORD"])
GPT_WAKE_WORD = "gpt"
LANGUAGE = os.environ["LANGUAGE"]
TRANSCRIBE = os.environ["TRANSCRIBE"]
TTS = os.environ["TTS"]


def extract_wake_word_and_user_command(user_phrase):
    bing_wake_word_indexes = []
    user_phrase_lower = user_phrase.lower()

    for wake_word in BING_WAKE_WORD:
        wake_word_lower = wake_word.lower()
        if wake_word_lower in user_phrase_lower:
            bing_wake_word_indexes.append(user_phrase_lower.find(wake_word_lower))

    # if bing wake word is found in any language
    if len(bing_wake_word_indexes) > 0:
        wake_word_index = max(bing_wake_word_indexes)
        user_phrase_from_wake_word = user_phrase[wake_word_index:]
        user_command = user_phrase_from_wake_word[user_phrase_from_wake_word.find(" "):].strip()
        return {"wake_word": BING_WAKE_WORD, "user_command": user_command}
    elif GPT_WAKE_WORD in user_phrase.lower():
        return {"wake_word": GPT_WAKE_WORD, "user_command": user_phrase}
    else:
        return None


def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


def has_english(text):
    return bool(re.search('[a-zA-Z]', text))


def has_hebrew(text):
    return any("\u0590" <= c <= "\u05EA" for c in text)


def has_language_neutral(text):
    return any(not c.isalnum() for c in text)


def detect_language(text):
    if has_cyrillic(text):
        return "ru"
    elif has_hebrew(text):
        return "he"
    if has_english(text):
        return "en"
    return None


def synthesize_speech(text, output_filename):
    if TTS == "GOOGLE":
        synthesize_speech_gtts(text, output_filename)
    else:
        synthesize_speech_poly(text, output_filename)


def synthesize_speech_gtts(text, output_filename):
    words_per_language = split_text_by_language(text)
    # delete file output_filename if it exists
    if os.path.exists(output_filename):
        os.remove(output_filename)
    previous_lang = None
    lang = "en"
    for word in words_per_language:
        if has_cyrillic(word):
            lang = "ru"
        if has_hebrew(word):
            lang = "iw"
        if has_language_neutral(word) and previous_lang is not None:
            lang = previous_lang
        gtts = gTTS(text=word, slow=False, lang=lang)
        # append to file
        with open(output_filename, 'ab') as ff:
            # if the test is just special symbol, it will throw exception
            try:
                gtts.write_to_fp(ff)
            except Exception as e:
                log.info(f"Exception in synthesize_speech_gtts for: {word}, error: {e}")
        previous_lang = lang

    # make the sound file two times faster
    sound = pydub.AudioSegment.from_file(output_filename, format="mp3")
    sound = sound.speedup(playback_speed=1.3)
    sound.export(output_filename, format="mp3")


def synthesize_speech_poly(text, output_filename):
    polly = boto3.client('polly', region_name='us-west-2')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli',
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())


# split string into multiple strings per language
def split_text_by_language_org(text):
    # split text into words
    words = text.split()
    words_per_language = []
    previous_lang = None
    for word in words:
        if has_cyrillic(word) and previous_lang == "ru":
            words_per_language[-1] += " " + word
        elif has_cyrillic(word):
            words_per_language.append(word)
            previous_lang = "ru"
        elif previous_lang == "en":
            words_per_language[-1] += " " + word
        else:
            # current language is not cyrillic and the previous language was None
            words_per_language.append(word)
            previous_lang = "en"
    return words_per_language


def split_text_by_language(text):
    # split text into words
    words = text.split()
    words_per_language = []
    previous_lang = None
    for word in words:
        language = detect_language(word)

        # new word, append it to the list
        if previous_lang is None:
            words_per_language.append(word)

        # same language as previous word, append it to the previous word
        elif language == previous_lang:
            words_per_language[-1] += " " + word

        # different language than previous word, append it to the list
        elif language != previous_lang:
            words_per_language.append(word)

        previous_lang = language

    return words_per_language


def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)


async def main():
    model = None
    if TRANSCRIBE != "GOOGLE":
        model = whisper.load_model("large-v2")
        log.info("whisper model loaded.")
    log.info("Initializing Bing Chatbot...")
    cookies = json.loads(open("cookies.json", encoding="utf-8").read())  # might omit cookies option
    bot = await Chatbot.create(cookies=cookies)
    last_conversation_time = time.time() - 60 * 60  # one hour back
    while True:

        try:
            with sr.Microphone() as source:
                sr.pause_threshold = 2
                recognizer.adjust_for_ambient_noise(source)

                waiting_for_wake_word = False
                if time.time() - last_conversation_time > 60:
                    waiting_for_wake_word = True
                    log.info(f"Waiting for wake words 'ok bing' or 'ok chat'...")

                audio = recognizer.listen(source, phrase_time_limit=15)
                log.info(f"Audio stream received. Now to recognize it...")
                phrase = transcribe(audio=audio, model=model, language=LANGUAGE)
                log.info(f"User said: {phrase}")

            if waiting_for_wake_word:
                user_command = extract_wake_word_and_user_command(phrase)
                if user_command is not None:
                    # if we have a wake word, start the conversation with the phrase after the space after  wake word
                    user_input = user_command["user_command"]
                    wake_word = user_command["wake_word"]
                    log.info(f"User input: {user_input}")
                else:
                    log.info("Not a wake word. Try again.")
                    continue
            else:
                user_input = phrase

            last_conversation_time = time.time()

            if wake_word == BING_WAKE_WORD:
                response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)

                # Select only the bot response from the response dictionary
                youtube_url = get_first_youtube_result(response["item"]["messages"])
                if youtube_url is not None:
                    log.info(f"Playing youtube video: {youtube_url}")
                    synthesize_speech(youtube_url['text'], 'response.mp3')
                    play_audio('response.mp3')
                    await bot.close()
                    log.info(f"Playing youtube video: {youtube_url['url']}")
                    play_youtube_video(youtube_url['url'])
                    continue
                for message in response["item"]["messages"]:
                    log.info(message)
                    if message["author"] == "bot":
                        bot_response = message["text"]
                # Remove [^#^] citations in response
                bot_response = response if bot_response is None else re.sub('\[\^\d+\^\]', '', bot_response)
            else:
                # Send prompt to GPT-3.5-turbo API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content":
                            "You are a helpful assistant."},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.5,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                    stop=["\nUser:"],
                )

                bot_response = response["choices"][0]["message"]["content"]
                last_conversation_time = time.time()

            log.info(f"Bot's response: {bot_response}")
            synthesize_speech(bot_response, 'response.mp3')
            play_audio('response.mp3')
            await bot.close()
        except Exception as error:
            log.info(f"Error: {error}")


def transcribe(audio, model, language):
    """
    Transcribe audio using the whisper model or google speech to text
    :param audio: audio to transcribe
    :param model: whisper model
    :param language: language to transcribe
    :return: transcription
    """

    if TRANSCRIBE == "GOOGLE":
        log.info("Transcribing audio with google...")
        phrase = recognizer.recognize_google(audio, language=LANGUAGE)
    else:
        log.info("Saving audio to audio.wav...")
        with open("audio.wav", "wb") as f:
            f.write(audio.get_wav_data())
        log.info("Transcribing audio with whisper...")
        result = model.transcribe("audio.wav", fp16=False, language=LANGUAGE, beam_size=1, best_of=1)
        phrase = result["text"]
    log.info(f"Transcription complete. You said: {phrase}")
    return phrase


def get_first_youtube_result(bing_responses):
    """
    Load the first youtube result from the bing responses
    :param bing_responses: bing responses
    :return: youtube video url
    """
    backup_url = None
    for response in bing_responses:
        log.info(f"analyzing response: {response}")
        if response['author'] == 'bot' and 'sourceAttributions' in response:
            for source in response['sourceAttributions']:
                if 'seeMoreUrl' in source and 'youtube.com/watch' in source['seeMoreUrl'] \
                        and 'channel' not in source['seeMoreUrl']:
                    url = source['seeMoreUrl']
                    response = requests.get(url)
                    print(f"Checking youtube url {url} status: {response.status_code}")
                    if response.status_code != 200:
                        continue
                    if 'text' in source and 'youtube' in source['text']:
                        return {"url": url, "text": source['providerDisplayName']}
                    return {"url": url, "text": 'youtube'}

    return backup_url


def play_youtube_video(url):
    """
    Play the youtube video in browser
    :param url: youtube video url
    :return: None
    """
    webbrowser.open(url)


if __name__ == "__main__":
    asyncio.run(main())
