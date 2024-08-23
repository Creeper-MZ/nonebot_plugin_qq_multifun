import base64
import io
import json
import os
import urllib
import urllib.request
import urllib.parse
import uuid
import random
import zipfile

import requests
import websocket
from io import BytesIO
from PIL import Image
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from .config import *

os.environ["PINECONE_API_KEY"] = CONFIG["PINECONE_API_KEY"]


class OllamaRam():
    headers = {"Content-Type": "application/json"}
    model = OLLAMA_CHAT_MODEL
    prompt_gen_model = OLLAMA_PROMPT_MODEL
    messageDIR = {}
    system_prompts = {}
    tts_url = CONFIG["TTS_CONFIG"]
    chat_url = CONFIG["OLLAMA_CHAT_URL"]
    prompt_gen_url = CONFIG["OLLAMA_PROMPT_URL"]
    ollama_prompt_gen_prompt_DEV = OLLAMA_PROMPT_GEN_DEV
    ollama_prompt_gen_prompt_DEV2 = OLLAMA_PROMPT_GEN_DEV2
    ollama_prompt_gen_prompt_nai = OLLAMA_PROMPT_GEN_DEV_FORNAI
    ollama_prompt_gen_prompt_nai2 = OLLAMA_PROMPT_GEN_DEV_FORNAI2
    global_system_prompts = ""
    options = OLLAMA_OPTIONS
    sd_webui_option = SD_OPTIONS
    sd_webui_url = CONFIG["SD_WEBUI_URL"]
    embedding = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')

    def __init__(self, global_system_prompt, url=chat_url):
        self.chat_url = url
        self.global_system_prompts = global_system_prompt

    def search_from_database(self, query):
        # query=self.embedding.embed_query(query)
        docs_with_scores = self.vectordb.max_marginal_relevance_search(query, k=120, fetch_k=120)
        result = []
        for doc in docs_with_scores:
            result += [doc.page_content.split(' - ')[1]]
        result = set(result)
        print("获取到：", len(result), "结果: \n")
        fin_result = ""
        for doc in result:
            fin_result += ',' + doc
        print(fin_result)
        return fin_result

    def load_database(self):
        self.vectordb = PineconeVectorStore(index_name=CONFIG["PINECONE_INDEX"], embedding=self.embedding)

    def add_texts_to_database(self, texts):
        self.vectordb.add_texts(texts=texts)

    def add_docs_to_database(self, doc_dir):
        docs = TextLoader(doc_dir, encoding="utf-8").load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=50, chunk_overlap=10)
        docs = text_splitter.split_documents(docs)
        self.vectordb.add_documents(documents=docs)

    def init_database(self, doc_dir):
        docs = TextLoader(doc_dir, encoding="utf-8").load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=50, chunk_overlap=10)
        docs = text_splitter.split_documents(docs)
        self.vectordb = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=self.embedding,
            index_name=CONFIG["PINECONE_INDEX"]
        )

    def add_user(self, username, system_prompt=""):
        if username in self.messageDIR:
            return False
        if (system_prompt == ""):
            self.system_prompts[username] = self.global_system_prompts
        else:
            self.system_prompts[username] = system_prompt
        self.messageDIR[username] = [{"role": "system", "content": self.system_prompts[username]}]
        return True

    def delete_user(self, username):
        if username not in self.messageDIR:
            return False
        del self.messageDIR[username]
        del self.system_prompts[username]
        return True

    def clear_user_message(self, username):
        if username not in self.messageDIR:
            return False
        del self.messageDIR[username]
        self.add_user(username, self.system_prompts[username])
        return True

    def set_system_prompt(self, username, system_prompt):
        if username not in self.messageDIR:
            return False
        self.system_prompts[username] = system_prompt
        self.messageDIR[username][0]["content"] = system_prompt
        return True

    def set_model(self, model):
        self.model = model

    def tts_trans(self, message):
        response = requests.get(self.tts_url + message)
        return response.content

    def to_base64(self, msg):
        base64_str = "base64://" + base64.b64encode(msg).decode('utf-8')
        return base64_str

    async def auto_prompt(self, message):
        prompt = [{"role": "system", "content": self.ollama_prompt_gen_prompt_DEV}]
        prompt.append({"role": "user", "content": "这些是一些你可以使用的prompt: " + self.search_from_database(
            message) + '\n' + self.ollama_prompt_gen_prompt_DEV2 + '\n这是生成prompt需要参考的信息: ' + message})
        data = {"model": self.model, "messages": prompt, "stream": False,
                "options": {"num_predict": 1024, "seed": random.randint(5, 100), "num_ctx": 10240 + 2048 * 3,
                            "num_batch": 128,
                            "num_keep": 24, "temperature": 0.8, "top_k": 20, "top_p": 0.95}}
        response = requests.post(self.chat_url, data=json.dumps(data), headers=self.headers)
        data = json.loads(response.text)
        ollama_response_msg = data
        print(ollama_response_msg['message']['content'])
        if ('(DEV_OUT)' not in ollama_response_msg['message']['content']):
            temp = ollama_response_msg['message']['content'].split('（DEV_OUT）')
        else:
            temp = ollama_response_msg['message']['content'].split('(DEV_OUT)')
        print("最终提示词: " + temp[1])
        return temp[1]

    async def auto_prompt_nai(self, message):
        prompt = [{"role": "system", "content": self.ollama_prompt_gen_prompt_nai}]
        prompt.append({"role": "user", "content": "这些是一些你可以使用的prompt: " + self.search_from_database(
            message) + '\n' + self.ollama_prompt_gen_prompt_nai2 + '\n这是生成prompt需要参考的信息: ' + message})
        data = {"model": self.model, "messages": prompt, "stream": False,
                "options": {"num_predict": 1024, "seed": random.randint(5, 100), "num_ctx": 10240 + 2048 * 3,
                            "num_batch": 128,
                            "num_keep": 24, "temperature": 0.8, "top_k": 20, "top_p": 0.95}}
        response = requests.post(self.chat_url, data=json.dumps(data), headers=self.headers)
        data = json.loads(response.text)
        ollama_response_msg = data
        print(ollama_response_msg['message']['content'])
        if ('(DEV_OUT)' not in ollama_response_msg['message']['content']):
            temp = ollama_response_msg['message']['content'].split('（DEV_OUT）')
        else:
            temp = ollama_response_msg['message']['content'].split('(DEV_OUT)')
        print("最终提示词: " + temp[1])
        return temp[1]

    def get_request(self, username, prompt):
        if username not in self.messageDIR:
            return False
        self.messageDIR[username].append({"role": "user", "content": prompt})
        data = {"model": self.model, "messages": self.messageDIR[username], "stream": False,
                "options": self.options}
        response = requests.post(self.chat_url, data=json.dumps(data), headers=self.headers)
        data = json.loads(response.text)
        ollama_response_msg = data['message']['content'].split('\n')
        final_msg = ''
        for msg in ollama_response_msg:
            if msg != '':
                final_msg += msg
        self.messageDIR[username].append({"role": "assistant", "content": final_msg})
        return final_msg

    async def get_sd_request_with_llama(self, prompt):
        temp_opt = self.sd_webui_option.copy()
        temp_opt["prompt"] += ',' + self.auto_prompt(prompt)
        print("发送绘画请求：: ", temp_opt)
        response = requests.post(url=self.sd_webui_url, json=temp_opt)
        return json.loads(response.text)["images"][0]

    async def get_sd_request(self, prompt):
        temp_opt = self.sd_webui_option.copy()
        temp_opt[
            "prompt"] = "score_9,score_8_up,score_7_up,source_anime BREAK,zPDXL2,<lora:cnv3mdde878c738thn20:0.8>,<lora:naipf:0.8>," + prompt
        response = requests.post(url=self.sd_webui_url, json=temp_opt)
        return json.loads(response.text)["images"][0]


class NovelAIAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = CONFIG["NOVELAI_URL"]
        self.headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def generate_image(
            self,
            input_text,
            model=NOVELAI_MODEL,
            width=832,
            height=1216,
            scale=6,
            sampler=NOVELAI_SAMPLER,
            steps=28,
            seed=None,
            negative_prompt=NOVELAI_NEGATIVE_PROMPT
    ):
        url = f"{self.base_url}/generate-image"
        data = {
            "input": input_text + ",best quality, amazing quality, very aesthetic, absurdres",
            "model": model,
            "action": "generate",
            "parameters": {
                "params_version": 1,
                "width": width,
                "height": height,
                "scale": scale,
                "sampler": sampler,
                "steps": steps,
                "n_samples": 1,
                "ucPreset": 0,
                "qualityToggle": True,
                "sm": True,
                "sm_dyn": True,
                "dynamic_thresholding": False,
                "controlnet_strength": 1,
                "legacy": False,
                "add_original_image": True,
                "cfg_rescale": 0,
                "noise_schedule": "native",
                "legacy_v3_extend": False,
                "seed": seed,
                "negative_prompt": negative_prompt,
                "reference_image_multiple": [],
                "reference_information_extracted_multiple": [],
                "reference_strength_multiple": [],
            },
        }

        data_json = json.dumps(data)
        print("NAI3开始绘画：" + input_text)
        response = requests.post(url, headers=self.headers, data=data_json)

        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                with zip_ref.open("image_0.png") as image_file:
                    image_data = image_file.read()
                    return io.BytesIO(image_data)
        else:
            raise Exception(
                f"Image generation failed with status code: {response.status_code}, message: {response.text}"
            )


class Comfy_API:
    model_name = COMFY_MODEL_NAMES

    def __init__(self):
        self.prompt_text = COMFY_REQUEST_JSON
        self.api_url = CONFIG["COMFY_URL"]
        self.client_id = str(uuid.uuid4())
        self.data = json.loads(self.prompt_text)
        # set the text prompt for our positive CLIPTextEncode
        self.data["prompt"]["183"]["inputs"]["positive"] = "1girl"
        self.data["client_id"] = self.client_id
        # set the seed for our KSampler node
        self.data["prompt"]["48"]["inputs"]["noise_seed"] = -1

    def queue_prompt(self, prompt):
        p = prompt
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request("http://{}/prompt".format(self.api_url), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    async def async_queue_prompt(self, prompt):
        p = prompt
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request("http://{}/prompt".format(self.api_url), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.api_url, url_values)) as response:
            return response.read()

    async def async_get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.api_url, url_values)) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.api_url, prompt_id)) as response:
            return json.loads(response.read())

    async def async_get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.api_url, prompt_id)) as response:
            return json.loads(response.read())

    def get_images(self, ws, prompt):
        print("Comfy开始绘画：" + prompt)
        self.data["prompt"]["183"]["inputs"]["positive"] = prompt
        self.data["prompt"]["183"]["inputs"]["negative"] = COMFY_NEGATIVE_PROMPT
        prompt_id = self.queue_prompt(self.data)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data
        print("Queue Done")
        history = self.get_history(prompt_id)[prompt_id]
        print(history)
        for o in history['outputs']:
            print(o)
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                output_images[node_id] = images_output
        return output_images

    async def get_images(self, ws, prompt):
        print("Comfy开始绘画：" + prompt)
        self.data["prompt"]["183"]["inputs"]["positive"] = prompt
        self.data["prompt"]["183"]["inputs"]["negative"] = COMFY_NEGATIVE_PROMPT
        prompt_id = self.queue_prompt(self.data)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data
        print("Queue Done")
        history = self.get_history(prompt_id)[prompt_id]
        print(history)
        for o in history['outputs']:
            print(o)
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                output_images[node_id] = images_output
        return output_images

    async def get_comfy_request(self, prompt):
        self.data["prompt"]["48"]["inputs"]["noise_seed"] = random.randint(0, 999999999999999)
        send_prompt = COMFY_BASE_POSITIVE_PROMPT + prompt
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.api_url, self.client_id))
        images = self.get_images(ws, send_prompt)
        import io
        image = io.BytesIO(images["48"][0])
        return image

    async def async_get_comfy_request(self, prompt):
        self.data["prompt"]["48"]["inputs"]["noise_seed"] = random.randint(0, 999999999999999)
        send_prompt = COMFY_BASE_POSITIVE_PROMPT + prompt
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.api_url, self.client_id))
        images = self.get_images(ws, send_prompt)
        import io
        image = io.BytesIO(images["48"][0])
        return image


# Initialize services
ollama = OllamaRam(global_system_prompt=SYSTEM_PROMPTS["Nahida"])
novel_ai = NovelAIAPI(CONFIG["NOVELAI_API_KEY"])
comfy_api = Comfy_API()
ollama.load_database()