# LLM Engineering
I used these projects to learn more about LLM engineering. They're in completion order.

Local models are on a VirtualBox VM with 16GB RAM and 8 CPUs.

## website_summarizer [🔗](website_summarizer)
- Use OpenAI API to summarize web page content and return the summary in Markdown.
- Do the same with a local version of llama3.2 and deepseek-r1:1.5b.
- Responses are saved by model name.
    - [deepseek-r1:1.5b-response.md](website_summarizer/deepseek-r1:1.5b-response.md)
    - [gpt-4o-mini-response.md](website_summarizer/gpt-4o-mini-response.md)
    - [llama3.2-response.md](website_summarizer/llama3.2-response.md)

Local models spiked VM CPU 100% causing desktop fans to come on. Lots of resources for a seemingly simple task, although the URL used does produce around 12k characters.

## brochure_builder [🔗](brochure_builder)
- Use OpenAI API to get base web page content and links
- Send links via API asking to decide which links are relevant for a brochure
- Get page contents for each relevant link
- Send all page content back to API and build a brochure
- Output saved in [brochure.md](brochure_builder/brochure.md)

This would be better implemented in something like CrewAI, but I wanted to do it the hard way first.

## multi_model_conversation [🔗](multi_model_conversation)
- Created a 3-way conversation between models
    - gpt-4o-mini is an argumentative, snarky bot
    - claude-3-5-haiku is the polite bot
    - gemini-2.0-flash-exp is a couples counselor impersonating Sigmond Freud
- Output saved in [conversation.txt](multi_model_conversation/conversation.txt)

## gradio_llm_chat [🔗](gradio_llm_chat)
- Basic GPT chat using Gradio

## basic_tool_call [🔗](basic_tool_call)
- Built on **gradio_llm_chat** to mock a basic airline flight pricing chatbot.
- Handles "tool_calls"

I can now see why frameworks like CrewAI are popular - it's a pain to code this.

## basic_rag [🔗](basic_rag)
- Uses files to add relevant context to the chat

## chroma_rag [🔗](chroma_rag)
- Expanded to use OpenAI embeddings fed to Chroma
- Visualize the vector store in 2D
- Add memory to the conversation via LangChain
- Uses Gradio for basic chat interface

## data_curation [🔗](data_curation)
Curate Amazon Review data to try and guess price based on descriptive text.
- uses Hugging Face McAuley-Lab/Amazon-Reviews-2023 dataset
- pulls in 8 categories from the dataset
- cleans data
- plots data to see distributions
- clean up distributions
- split data into training and test data
- upload dataset to HuggingFace
- pickle the datasets so we don't have to do all this when we use it again

## basic_model_training [🔗](basic_model_training)
Go through various models and see how well they do predicting price based on product text.
- uses pickled datasets from [data_curation](data_curation)
- tests against:
  - simple average price of items [📊](basic_model_training/average_pricer.png)
    - Error \$137.17
    - Hits 15.2%
  - Linear regression using Item Weight, Best Sellers Rank, Brand, is_top_brand features [📊](basic_model_training/linear_regression_pricer.png)
    - Error \$139.20
    - Hits 15.6%
  - bag-of-words text features + linear regression [📊](basic_model_training/bow_lr_pricer.png)
    - Error \$113.60
    - Hits 24.8%
  - Word2Vec [📊](basic_model_training/word2vec_lr_pricer.png)
    - Error \$113.14
    - Hits 22.8%
  - Linear SVR [📊](basic_model_training/svr_pricer.png)
    - Error \$110.91
    - Hits 29.2%
  - Random Forest Regressor [📊](basic_model_training/random_forest_pricer.png)
    - Error \$105.10
    - Hits 37.6%