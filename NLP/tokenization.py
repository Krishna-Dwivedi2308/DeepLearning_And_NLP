import tiktoken
encoder=tiktoken.encoding_for_model("gpt-4o")
print("Vocabulary size of gpt-3.5-turbo:", encoder.n_vocab)

tokens = encoder.encode("Hello, how are you?")
print("Tokens:", tokens)
print("Decoded text:", encoder.decode(tokens))