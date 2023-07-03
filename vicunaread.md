<img alt="camel" height="200" src="./docs/shennonglogo.png" width="200"/>

# **Shennong: A Pharmaceutical Chat Model via Unifying Heterogeneous Domain Knowledge**




## Directory

- [What do we do?](https://github.com/ShuoSun-CN/Shennong#what-do-we-do)


- [performance and case](https://github.com/ShuoSun-CN/Shennong#Performance-and-case)

[//]: # (  - **Checkpoint-4000**&#40;Facico/Chinese-Vicuna-lora-7b-0.75epoch-belle-and-guanaco&#41;)

[//]: # ()
[//]: # (  - **Checkpoint-8000**&#40;Facico/Chinese-Vicuna-lora-7b-1.5epoch-belle-and-guanaco&#41;)

[//]: # ()
[//]: # (  - **Checkpoint-final**&#40;Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco&#41; and it is used for multiple rounds of dialogue)
- [How to use](https://github.com/ShuoSun-CN/Shennong#how-to-use)
  - Installing、Multi-gpu training、Single-gpu training、Inference and use gradio to generate a web page(Streaming mode+beam search)、 multi-round interaction and use gradio to generate a web page(Streaming mode+beam search)、Streaming mode base on alpaca-lora-serve
- [try on colab](https://github.com/Facico/Chinese-Vicuna#try-on-colab)
## Overview

-  LLaMA paper: https://arxiv.org/abs/2302.13971v1
-  Self-Instruct paper: https://arxiv.org/abs/2212.10560
-  data generation: https://github.com/LianjiaTech/BELLE and https://guanaco-model.github.io/
-  the first work: https://github.com/tatsu-lab/stanford_alpaca

We currently select the combination of BELLE and Guanaco data as our main training dataset. 
We will train on multi-turn instruction data.

## What do we do?

Based on [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna), we use different kinds of data to finetune this model to get Shennong.
And Shennong is good at answering the question about pharmacology. It's magic that Shennong can not only answer the frontier question about 
pharmacology but also help predict the route of synthesising a molecule or modify molecule that it can have better character.

## Performance and Case

### Performance
This figure show the performance of Shennong and other large language models on data of Drugbank and article about
 pharmacology.


![img_1.png](./docs/performance.png)

[//]: # (**Checkpoint-8000**&#40;about 1.5 epoch on belle+guanaco&#41;)

[//]: # ()
[//]: # (| Human                                                        | Chinese-Vicuna                                               | Note                    |)

[//]: # (| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- |)

[//]: # (| 写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流 | A: Spring rain falls softly on my face,<br/>A cool breeze blows through my hair,<br/>The sun shines brightly in the sky,<br/>And I feel as if I am flying high.<br/><br/>The birds chirp happily in the trees,<br/>The flowers bloom and their scent fills the air,<br/>The bees buzz gently from flower to flower,<br/>And the world seems to come alive once more.<br/><br/>I can hear the rush of the river,<br/>The sound of the waterfalls,<br/>The splash of the raindrops on the rocks,<br/>And the gentle murmur of the stream.<br/><br/>My heart is filled with joy and peace,<br/>As I take in the beauty around me,<br/>The warmth of the sun on my skin,<br/>And the freshness of the spring air. | Repetition Penalty=2.2  |)

[//]: # (| 写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流 | 春雨滴滴滚滚，<br/>滚滚滚滚滚滚。<br/>滚滚滚滚滚滚，<br/>滚滚滚滚滚滚。<br/><br/>春雨滚滚滚，<br/>滚滚滚滚滚滚。<br/>滚滚滚滚滚滚，<br/>滚滚滚滚滚滚。<br/><br/>春雨滚滚滚，<br/>滚滚滚滚滚滚。<br/>滚滚滚滚滚滚，<br/>滚滚滚滚滚滚。<br/><br/>春雨滚滚滚，<br/>滚滚滚 | Repetition Penalty=1.0  |)

[//]: # (| Spring rain falls softly on my face,<br/>A cool breeze blows through my hair,<br/>The sun shines brightly in the sky,<br/>And I feel as if I am flying high.<br/><br/>The birds chirp happily in the trees,<br/>The flowers bloom and their scent fills the air,<br/>The bees buzz gently from flower to flower,<br/>And the world seems to come alive once more.<br/><br/>I can hear the rush of the river,<br/>The sound of the waterfalls,<br/>The splash of the raindrops on the rocks,<br/>And the gentle murmur of the stream.<br/><br/>My heart is filled with joy and peace,<br/>As I take in the beauty around me,<br/>The warmth of the sun on my skin,<br/>And the freshness of the spring air.<br/>把上面的散文诗翻译成中文 | 春雨轻轻洒落在我的脸上，<br/>轻轻的风轻轻吹过我的头发，<br/>阳光明媚地照耀在天空，<br/>我感觉自己在飞高。<br/><br/>鸟儿在树枝上唱歌，<br/>花朵开放，它们的香气充满了空气，<br/>蜜蜂在花朵间飞舞，<br/>它们的鸣声充满了温暖。<br/><br/>我听到河流的声音，<br/>水瀑在岩石上轻轻滴滴，<br/>雨水在石头上轻轻滴滴，<br/>它们的声音充满了宁静。<br/><br/>我的心充满了喜悦和平，<br/>我感受到热浪的阳光，<br/>温暖的空气洒在我的脸上，<br/>我感受到春季的美 | beams=2，max tokens=350 |)

[//]: # (| Classify the sentiment of the following sentence into Positive, Neutral, or Negative:<br/>Super excited about teaching Stanford’s first course on Large Language Models! Check the syllabus out here | Positive                                                     |                         |)

[//]: # (| 今天天气怎么样，把这句话翻译成英语                           | What's the weather like today?                               |                         |)

[//]: # (| 今天天气怎么样，把这句话翻译成日语                           | 今天の天気は何ですか？                                       |                         |)

[//]: # (| 明天就假期结束了，有点抗拒上班，应该什么办？                 | 明天就假期结束了，有点抗拒上班，应该什么办？可以考虑一些有趣的活动，比如去看电影、阅读书籍、听音乐、去游泳池或者参加户外活动等。此外，也可以考虑一些家庭活动，比如做烹饪、做玩具、做手工艺品等。 |                         |)

[//]: # (| Tell me five words that rhyme with 'shock'.                  | Five words that rhyme with'shock' are: <br/>1. Rock <br/>2. Lock <br/>3. Cock <br/>4. Hock <br/>5. Stock |                         |)

[//]: # ()
[//]: # (- We can find that you do not punish him a little, he will mess up, if the generation effect is not good, you can increase the Repetition Penalty, beams, topk.)

[//]: # (- If your input is long, we suggest you can turn down the beams)

[//]: # (- The checkpoint-8000 has some Japanese language skills, but not strong)
### Case
These figures show cases that Shennong answer the question of molecule synthesising and molecule modifying.


figure

figure


## How to use

**Installation**

```
git clone https://github.com/ShuoSun-CN/Shennong#
pip install -r requirements.txt
```

Local python environment is 3.8, torch is 1.13.1, CUDA is 12

NOTE: python3.11 has a known `torchrun` bug, details [here](https://github.com/facebookresearch/llama/issues/86)

**Multi-round interaction**

As we use the basic command prompt when training, so the ability of small talk conversation is still relatively poor, the follow-up will increase this part of the training.

```bash
python interaction.py --lora_path shennong-lora
```

- A simple interactive interface constructed using gradio, which allows you to set the max_memory according to your machine (it will intercept the max_memory part later in the history conversation)

- The prompt used in this script is not quite the same as the one used in generate.sh. The prompt in this script is in the form of a dialogue, as follows

  - ```
    The following is a conversation between an AI pharmacologist called Shennong and a human user called User.
    ```





**Shennong-web Installation**

Before we install Shennong-web service ,we need setup our development environment firstly.

***Development Environment Setup***

****Node****

`node` need `^16 || ^18` vision（`node >= 14`
need install [fetch polyfill](https://github.com/developit/unfetch#usage-as-a-polyfill)
），use [nvm](https://github.com/nvm-sh/nvm) can manage multiple local `node` visions

```shell
node -v
```

****PNPM****

If you haven't installed  `pnpm`, you should install it by this command.

```shell
npm install pnpm -g
```



After that, we can install Shennong on our own server by these commands.

***Back-end Service Installation***
```bash
python ./shennong-web/service/main.py --lora_path shennong_lora
```

- This command can switch on Shennong back-end service on specific port.
- You can modify the port, but you also modify the port of front-end service so that these two services can contact normally.

***Front-end Service Installation***
```bash
pnpm bootstrap
pnpm dev
```

- This command can switch on Shennong front-end service on specific port.
- Please run these command in root directory of this project.


[//]: # (**Single-gpu Training**)

[//]: # ()
[//]: # (```)

[//]: # (python finetune.py --data_path merge.json --test_size 2000)

[//]: # (```)

[//]: # ()
[//]: # (- The test_size cannot be larger than the data size)

[//]: # ()
[//]: # (**inference and use gradio to generate a web page**)

[//]: # ()
[//]: # (```bash)

[//]: # (bash generate.sh)

[//]: # (```)

[//]: # ()
[//]: # (- The parameters to note here are as follows)

[//]: # ()
[//]: # (  - BASE_MODEL，path of LLM)

[//]: # (  - LORA_PATH，The checkpoint folder of the lora model)

[//]: # (    - It should be noted here that the config loaded by the lora model must be "adapter_config.json" and the model name must be "adapter_model.bin", but it will be automatically saved as "pytorch_model.bin" during training. pytorch_model.bin" during training, while "adapter_config.json" and "adapter_model.bin" will be saved after all training is finished)

[//]: # (      - If you load the lora model in the training checkpoint, the code will automatically copy the local "config-sample/adapter_config.json" to the corresponding directory for you and rename the "pytorch_model.bin" to "adapter_model.bin". and rename "pytorch_model.bin" to "adapter_model.bin".)

[//]: # (    - It can also be any lora model on the huggingface corresponding to llama 7B, e.g.: `Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco`)

[//]: # (  - USE_LOCAL, which checks the local model configuration when set to 1)

[//]: # (- When using, "max_tokens" is set according to your computer's video memory, and if the generated content generates a lot of duplicate information, you can turn up the "Repetition Penalty".)

[//]: # ()

**Finetune Based on Shennong**

If you want to finetune Shennong with your own data, this command will help you.
```bash
python finetune.py --lora_path shennong-lora --data_path yourdata.json --output_path youroutpath
```

- `yourdata.json` is your own data.

- `youroutpath`  is the directory that you want to save the model parameters.

The data format is relatively simple, basically as follows, with simple examples such as:
```
{
'instruction': 
'input': 
'output'
}
```

## Try on colab
If you just want to have try or you don't have enough resource to build Shennong, you can click this link.
This link will help you build Shennong on Colab which cost nothing.

| colab link                                                   | Descriptions                       |
| ------------------------------------------------------------ |------------------------------------|
| [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1ftDqBVTRkADHPYn3ZFRfWXRbG83iGv4k?usp=sharing) | Multiple intercation with Shennong |




