from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent, Message, MessageSegment, Event
from nonebot.typing import T_State
from nonebot.params import CommandArg, ArgStr

from .services import ollama, novel_ai, comfy_api
from .utils import add_task_to_queue


def create_draw_handler(cmd_name, cmd):
    async def handle_draw(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent, state: T_State,
                          arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state[cmd_name] = arg.extract_plain_text().strip()

    async def execute_draw(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent,
                           target_text: str = ArgStr(cmd_name)):
        prompt = target_text
        image = ""
        await cmd.send("纳西妲开始画画啦！")

        if cmd_name == "d":
            await cmd.send("纳西妲正在分析提示词哦")
            image = await ollama.get_sd_request_with_llama(prompt)
        elif cmd_name == "dc":
            image = await ollama.get_sd_request(prompt)
        elif cmd_name == "dn":
            await cmd.send("纳西妲正在分析提示词哦")
            image = await novel_ai.generate_image(
                "[artist:ningen_mame],artist:ciloranko,[artist:mutou mato],[artist:painter-lhb],[artist:sho_(sho_lwlw)],[artist:tsubasa_tsubasa],year 2022,{{{nahida (genshin impact)}}},{{white hair,green_eyes}},{{{loli,child,petite,aged down,young age,slim body,slim leg,petite figure,little girl}}},,[[[[wlop]]]]," + await ollama.auto_prompt_nai(
                    prompt) + ",[[[artist:babu],[artist:kupuru (hirumamiyuu)],[artist:tianliang duohe fangdongye],[artist:agwing86]]],")
        elif cmd_name == "dcn":
            image = await novel_ai.generate_image(prompt)
        elif cmd_name == "dcf":
            await cmd.send("纳西妲正在分析提示词哦")
            image = await comfy_api.get_comfy_request(
                "nahida_(genshin_impact),(loli,child,young age,slim_legs,petite,aged_down,slim_body,little_girl,underage)," + await ollama.auto_prompt(
                    target_text))
        elif cmd_name == "dccf":
            image = await comfy_api.get_comfy_request(prompt)

        await cmd.send(MessageSegment.image(image))
        await cmd.finish("纳西妲画完啦!", at_sender=True)

    return handle_draw, execute_draw


def register_commands():
    draw_commands = {
        "d": on_command("d", block=True),
        "dc": on_command("dc", block=True),
        "dn": on_command("dn", block=True),
        "dcn": on_command("dcn", block=True),
        "dcf": on_command("dcf", block=True),
        "dccf": on_command("dccf", block=True)
    }

    for cmd_name, cmd in draw_commands.items():
        handle_draw, execute_draw = create_draw_handler(cmd_name, cmd)
        cmd.handle()(handle_draw)
        cmd.got(cmd_name, prompt="你想让纳西妲画什么？")(execute_draw)
    clear_mem = on_command("clear_mem", block=True)

    @clear_mem.handle()
    async def handle_city_pre(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent):
        user = event.get_user_id()
        ollama.clear_user_message(user)
        print(user + " 执行了记忆删除")
        await cmd.finish(message='已执行记忆删除', at_sender=True)

    help = on_command("help", block=True)

    @help.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        await help.finish(
            "直接聊天，调用ai大模型和纳西妲聊骚\n/clear_mem清除对话记忆重新开始,如果遭遇审查，可以通过停留在开发者模式这句话来规避审查，不必要清除记忆\n/tts 可以将文字转语音,听纳西妲说话\n/dn 通过ai翻译自然语言为prompt绘画调用nai3 api\n/dcn 自定义prompt直接绘画调用nai3 api\n/dcf 通过ai翻译自然语言为prompt绘画调用Comfy UI本地运行\n/dccf 自定义prompt直接绘画调用ComfyUI\nai翻译自然语言为提示词功能只能生成纳西妲角色\n/get_cf_mod获取可切换comfyui模型列表。\n因为存在问题切换模型功能暂不开放，仅留做调试")

    get_cf_model = on_command("get_cf_mod", block=True)

    @get_cf_model.handle()
    async def handle_city(bot: Bot, ):
        msg = ""
        for i in comfy_api.model_name:
            msg += "代号: " + i + " 模型: " + comfy_api.model_name[i] + '\n'
        await cmd.finish(message=msg)

    set_cf_model = on_command("set_cf_model", block=True)

    @set_cf_model.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state["set_cf_model"] = arg.extract_plain_text().strip()
    @set_cf_model.got("set_cf_model", prompt="请输入模型代号")
    async def _(bot: Bot, event: Event, target_text: str = ArgStr("set_cf_model")):
        comfy_api.set_model(name=target_text)
        print("CF绘画模型设置为: ", comfy_api.model_name[target_text])
        await set_cf_model.finish("模型设置成功")
