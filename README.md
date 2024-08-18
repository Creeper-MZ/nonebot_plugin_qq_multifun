# QQ_DrawBot
 一个QQ的对话，绘画，语音生成机器人，能使用Ollama，Comfyui，sdwebui，NAI3
### 参考了B站大佬: 
@啥都会一点的鼠鼠 的机器人代码。我自己是个小白，代码是边谷歌边写的，所以很乱，还望包含。
### 推荐使用: Napcat 使用反向ws连接
[NapCat](https://github.com/NapNeko/NapCatQQ)
#### Napcat的一些参数如下(这里只列出需要修改的部分，别的保持默认就好)：
```json
"http":{
    "enable":true,
    "host"："0.0.0.0",
    "prot":25533,
    },
"ws":{
    "enable":true,
    "host":"0.0.0.0",
    "port":25522
    },
"reverseWs":{
    "enable":true,
    "urls":["ws://127.0.0.1:8080/onebot/v11/ws"]
    },
```
### 一些小事项：
#### 需要自己准备nai3和的api_key放到代码里
#### 向量数据库使用了Pinecone，用来提高AI生成提示词的准确度，免费的，但是需要自己加载提示词
#### 向量数据库除了要定义API，还要定义Index
#### 代码里的各种提示词都挺乱的，估计会让人头大
#### 推荐Ollama 使用model为gemma2:27b,mistral-nemo:12b,当然，有别的不错的也可以用
### 以下的api地址需要自己输入
tts(GPT-SoVits),ComfyUI,sd-Web-UI,Ollama的API地址都需要自己修改