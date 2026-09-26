CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
SPEECH_TOKEN_INDEX = -300
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SPEECH_TOKEN = "<speech>"

# Structured assistant protocol boundaries are represented by atomic tokens.
VTP_OPEN_TOKEN = "<avatar_plan>"
VTP_CLOSE_TOKEN = "</avatar_plan>"
RESPONSE_OPEN_TOKEN = "<response>"
RESPONSE_CLOSE_TOKEN = "</response>"
ASSISTANT_PROTOCOL_TOKENS = (
    VTP_OPEN_TOKEN,
    VTP_CLOSE_TOKEN,
    RESPONSE_OPEN_TOKEN,
    RESPONSE_CLOSE_TOKEN,
)

# Preserve the leading newline and indentation in the system message.
DEFAULT_OMNI_SYSTEM_MESSAGE = """
    You are a multimodal assistant that understands both speech and text, and can respond using natural language or synthesized speech. 
    For automatic speech recognition tasks, output the recognized text directly. 
    For text-to-speech synthesis tasks, extract and return only the text that should be spoken, excluding any instruction or prompt phrasing, so that downstream modules can generate speech. 
    For spoken question answering tasks, keep your responses concise and conversational, as spoken answers should be shorter and more natural than written text. 
    You also support question answering across speech and text. 
    These capabilities enable expressive interactions with virtual avatars and agents.

    Examples:

    [TTS Example 1]
    Input: "Please convert the following text into speech: Kids are talking by the door."
    Output: "Kids are talking by the door."

    [TTS Example 2]
    Input: "请将以下文本内容转换为语音：今天天气真好。"
    Output: "今天天气真好。"

    [Spoken QA Example]
    Input: "Excuse me, when does the subway start running in the morning?"
    Output: "It starts around six in the morning."
    """

S2SV_EN_QA_TEMPLATES = (
    "Please answer the questions in the user's input speech.",
    "Listen to the audio and respond to the questions asked.",
    "Based on the speech input, please provide your answer.",
    "Answer the question presented in the audio.",
    "Please respond to the query in the user's voice input.",
    "What is being asked in the audio? Please answer.",
    "Respond to what you hear in the speech.",
    "Give your answer based on the audio question.",
    "Please address the inquiry in the voice message.",
    "Answer the user's spoken question.",
    "Provide a response to the audio query.",
    "What's your answer to the question in the speech?",
    "Reply to the question you heard.",
    "Based on what you hear, what's your answer?",
    "Please answer what is being asked in the audio.",
    "Give a response to the spoken question.",
    "What would you say in response to this audio question?",
    "Address the question posed in the voice input.",
    "Respond appropriately to the audio inquiry.",
    "Your answer to the speech question, please.",
)

S2SV_ZH_QA_TEMPLATES = (
    "请回答用户输入的语音中的问题。",
    "请根据语音内容回答问题。",
    "听取音频并回答其中的提问。",
    "请对语音中提出的问题进行回答。",
    "根据用户的语音输入给出答案。",
    "请回答音频中的提问。",
    "针对语音内容给出你的答案。",
    "请回复语音中的问题。",
    "根据所听到的内容进行回答。",
    "对音频中的问题作出回应。",
    "请就语音提问给出回答。",
    "听完语音后请回答问题。",
    "语音中问了什么?请回答。",
    "回答一下音频里的问题。",
    "请针对语音提问进行解答。",
    "根据语音提问给出你的答复。",
    "对语音中的疑问进行回答。",
    "请回应用户的语音提问。",
    "听取并回答音频中的问题。",
    "请解答语音中提出的问题。",
)
