# import os
# import torch
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory

# def get_conversation_chain(vector_store):
#     # Memory setup
#     memory = ConversationBufferMemory(
#         memory_key='chat_history',
#         return_messages=True
#     )
    
#     # Model setup with quantization and CPU offloading
#     model_name = "mistralai/Mistral-7B-Instruct-v0.3"
#     token = os.environ.get("HUGGINGFACE_TOKEN")

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         llm_int8_enable_fp32_cpu_offload=True
#     )
    
#     # Let HuggingFace handle device placement
#     device_map = "auto"
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map=device_map,
#         torch_dtype=torch.bfloat16,
#         token=token
#     )
    
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         temperature=0.7,
#         do_sample=True
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)

#     contextualize_q_prompt = ChatPromptTemplate.from_messages([
#         ("system", "Reformulate questions considering chat history."),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ])
    
#     qa_prompt = ChatPromptTemplate.from_messages([
#         ("system", (
#             "You are a research assistant. Always answer using the context below. "
#             "If the answer isn't explicit, make an educated guess. "
#             "Format answers with bullet points and key terms in bold.\n\n"
#             "Context:\n{context}"
#         )),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ])

#     retriever = vector_store.as_retriever(search_kwargs={"k": 5})
#     history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
#     return create_retrieval_chain(history_aware_retriever, question_answer_chain), memory
