import requests
from transformers import AutoTokenizer  # type: ignore


class Assistant:
    """
    Generic Assistant LLM api wrapper (vLLM)
    """

    def __init__(
        self,
        endpoint: str = "",  # vLLM endpoint
        max_seq_len: int = 32768,
        tokenizer: str = "mistralai/Mixtral-8x7B-v0.1",
        system_prompt: str | None = "",
        system_prefix: str = "",
        use_system_prompt_in_user_prompt: bool = True,
        system_prompt_in_user_prompt_separator: str = "",
        user_prefix: str = "",
        assistant_prefix: str = "",
        use_short_term_memory: bool = False,
        short_term_memory_max_tokens: int = 1024,
        debug: bool = False,
    ) -> None:
        """
        Initialize the AI Assistant

        Args:
            endpoint (str): The endpoint to use for the assistant (vLLM)
            max_seq_len (int): The maximum sequence length to use
            tokenizer (str): The tokenizer to use (huggingface model name)
            system_prompt (str|None): The system prompt to use. Set to None to disable.
            system_prefix (str): The system prefix to use
            use_system_prompt_in_user_prompt (bool): Whether to use the system prompt
                                                     in the user prompt
            system_prompt_in_user_prompt_separator (str): The separator to use between
                                                    the system prompt and user prompt
            user_prefix (str): The user prefix to use
            assistant_prefix (str): The assistant prefix to use
            use_short_term_memory (bool): Whether to use short term memory
            short_term_memory_max_tokens (int): The maximum number of tokens to store
                                                 in short term memory
            debug (bool): Whether to enable debug mode
        """
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.max_seq_len = max_seq_len
        self.system_prefix = system_prefix
        self.use_system_prompt_in_user_prompt = use_system_prompt_in_user_prompt
        self.system_prompt_in_user_prompt_separator = (
            system_prompt_in_user_prompt_separator
        )
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.debug = debug

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.use_short_term_memory = use_short_term_memory
        self.short_term_memory = []  # type: list
        self.short_term_memory_max_tokens = short_term_memory_max_tokens

    def _construct_prompt(
        self,
        user_prompt: str,
        inject_messages: list | None = None,
        truncate_to: int | None = None,
        truncate_style: str = "right",
        truncate_append: str = "",
    ) -> str:
        """
        Construct the prompt for the chat completion

        Args:
            user_prompt (str): The current user prompt
            inject_messages (list): The messages to inject into the prompt
                                    in the format of [
                                        {
                                            "role": "user",
                                            "content": "User message to inject",
                                        },
                                        {
                                            "role": "assistant",
                                            "content": "Assistant message to inject",
                                        },
                                    ]
            truncate_to (int|None): The number of tokens to truncate the prompt to
            truncate_style (str): The style to truncate the prompt with (left or right)
            truncate_append (str): The string to append to the prompt after truncation

        Returns:
            str: The prompt for the chat completion
        """
        if inject_messages is None:
            inject_messages = []

        prompt = ""
        if self.system_prompt is not None:
            if self.use_system_prompt_in_user_prompt:
                user_prompt = (
                    self.system_prompt
                    + self.system_prompt_in_user_prompt_separator
                    + user_prompt
                )
            else:
                prompt += self.system_prefix + self.system_prompt

        if self.use_short_term_memory:
            for message in self.short_term_memory:
                prefix = (
                    self.user_prefix
                    if message["role"] == "user"
                    else self.assistant_prefix
                )
                prompt += prefix + message["content"]

        for message in inject_messages:
            prefix = (
                self.user_prefix if message["role"] == "user" else self.assistant_prefix
            )
            prompt += prefix + message["content"]

        truncate_value = truncate_to or self.max_seq_len - 300
        prompt_tokens = self.tokenizer.encode(user_prompt, add_special_tokens=False)
        if truncate_style == "left":
            prompt_tokens = prompt_tokens[-truncate_value:]
        elif truncate_style == "right":
            prompt_tokens = prompt_tokens[:truncate_value]

        prompt += self.user_prefix + self.tokenizer.decode(
            prompt_tokens, skip_special_tokens=True
        )
        prompt += truncate_append + self.assistant_prefix
        if self.debug:
            print("Prompt:", prompt)
        return prompt

    def calculate_num_tokens(self, text: str, add_special_tokens=False) -> int:
        """
        Calculate the number of tokens in a given text

        Args:
            text (str): The text to calculate the number of tokens for
            add_special_tokens (bool): Whether to add special tokens

        Returns:
            int: The number of tokens in the text
        """
        return len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def calculate_short_term_memory_tokens(self) -> int:
        """
        Calculate the number of tokens in short term memory

        Returns:
            int: The number of tokens in short term memory
        """
        total = 0
        for message in self.short_term_memory:
            prefix = (
                self.user_prefix if message["role"] == "user" else self.assistant_prefix
            )
            total += self.calculate_num_tokens(prefix + message["content"])
        return total

    def add_message_to_short_term_memory(self, message: dict) -> None:
        """
        Add a message to short term memory

        Args:
            message (dict): The message to add to short term memory
        """
        self.short_term_memory.append(message)
        while (
            self.calculate_short_term_memory_tokens()
            > self.short_term_memory_max_tokens
        ):
            self.short_term_memory.pop(0)  # Remove the oldest message

    def get_response(
        self,
        user_prompt: str,
        inject_messages: list | None = None,
        truncate_to: int | None = None,
        truncate_style: str = "right",
        truncate_append: str = "",
        n: int = 1,
        use_beam_search: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_tokens: int | None = None,
        stop: list | None = None,
        logit_bias: dict | None = None,
        retries: int = 3,
        stream: bool = False,
        timeout: int = 600,
        endpoint: str | None = None,
    ) -> str:
        """
        Get a chat response from the model

        Args:
            user_prompt (str): The current user prompt
            inject_messages (list): The messages to inject into the prompt
                                    in the format of [
                                        {
                                            "role": "user",
                                            "content": "User message to inject",
                                        },
                                        {
                                            "role": "assistant",
                                            "content": "Assistant message to inject",
                                        },
                                    ]
            truncate_to (int|None): The number of tokens to truncate the prompt to
            truncate_style (str): The style to truncate the prompt with (left or right)
            truncate_append (str): The string to append to the prompt after truncation
            n (int): The number of responses to generate
            use_beam_search (bool): Whether to use beam search
            temperature (float): The temperature to use
            top_p (float): The top p to use
            top_k (int): The top k to use
            presence_penalty (float): The presence penalty to use
            frequency_penalty (float): The frequency penalty to use
            max_tokens (int): The maximum number of tokens to generate
            stop (list): The stop tokens to use
            retries (int): The number of retries to use
            stream (bool): Whether to stream the response
            timeout (int): The timeout to use
            endpoint (str|None): The endpoint to use (overrides self.endpoint)

        Returns:
            str: The assistant's response
        """
        try:
            if inject_messages is None:
                inject_messages = []
            if stop is None:
                stop = []
            if logit_bias is None:
                logit_bias = {}
            if max_tokens is None:
                max_tokens = self.max_seq_len
            if stop is None:
                stop = []
            prompt = self._construct_prompt(
                user_prompt,
                inject_messages,
                truncate_to,
                truncate_style,
                truncate_append,
            )

            headers = {"User-Agent": "Assistant Client"}
            pload = {
                "prompt": prompt,
                "n": n,
                "use_beam_search": use_beam_search,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "stop": stop,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            endpoint = endpoint or self.endpoint
            response = requests.post(
                endpoint, headers=headers, json=pload, stream=True, timeout=timeout
            ).json()["text"][0][len(prompt) :]

            if self.use_short_term_memory:
                self.add_message_to_short_term_memory(
                    {"role": "user", "content": user_prompt}
                )
                self.add_message_to_short_term_memory(
                    {"role": "assistant", "content": response}
                )
            return response
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            if retries > 0:
                return self.get_response(
                    user_prompt=user_prompt,
                    inject_messages=inject_messages,
                    truncate_to=truncate_to,
                    truncate_style=truncate_style,
                    truncate_append=truncate_append,
                    n=n,
                    use_beam_search=use_beam_search,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    max_tokens=max_tokens,
                    stop=stop,
                    retries=retries - 1,
                    stream=stream,
                )
            raise e
