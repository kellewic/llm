# LLM Engineering
I used these projects to learn more about LLM engineering. They're in completion order.

Local models are on a VirtualBox VM with 24GB RAM and 16 CPUs. I started with 16GB RAM and 8 CPUs, but it struggled a lot.

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
    - Error=\$137.17, RMSLE=1.19, Hits=15.2%
  - Linear regression using Item Weight, Best Sellers Rank, Brand, is_top_brand features [📊](basic_model_training/linear_regression_pricer.png)
    - Error=\$139.20, RMSLE=1.17, Hits=15.6%
  - bag-of-words text features + linear regression [📊](basic_model_training/bow_lr_pricer.png)
    - Error=\$113.60, RMSLE=0.99, Hits=24.8%
  - Word2Vec [📊](basic_model_training/word2vec_lr_pricer.png)
    - Error=\$113.14, RMSLE=1.05, Hits=22.8%
  - Linear SVR [📊](basic_model_training/svr_pricer.png)
    - Error=\$110.91, RMSLE=0.92, Hits=29.2%
  - Random Forest Regressor [📊](basic_model_training/random_forest_pricer.png)
    - Error=\$105.10, RMSLE=0.89, Hits=37.6%

## frontier_model_test [🔗](frontier_model_test)
- uses pickled test dataset from [data_curation](data_curation)
- tests against:
  - gemini-2.0-flash [📊](frontier_model_test/gemini-2.0-flash.png)
    - Error=\$73.48, RMSLE=0.56, Hits=56.4%
  - gpt-4o-2024-08-06 [📊](frontier_model_test/gpt-4o-2024-08-06.png)
    - Error=\$75.66, RMSLE=0.89, Hits=57.6%
  - gemini-2.0-flash-lite [📊](frontier_model_test/gemini-2.0-flash-lite.png)
    - Error=\$76.42, RMSLE=0.61, Hits=56.0%
  - gpt-4o-mini [📊](frontier_model_test/gpt-4o-mini.png)
    - Error=\$81.61, RMSLE=0.60, Hits=51.6%
  - claude-3-5-haiku-20241022 [📊](frontier_model_test/claude-3-5-haiku-20241022.png)
    - Error=\$85.25, RMSLE=0.62, Hits=50.8%
  - claude-3-5-sonnet-20241022 [📊](frontier_model_test/claude-3-5-sonnet-20241022.png)
    - Error=\$88.97, RMSLE=0.61, Hits=49.2%
  - claude-3-7-sonnet-20250219 [📊](frontier_model_test/claude-3-7-sonnet-20250219.png)
    - Error=\$89.41, RMSLE=0.62, Hits=55.2%
  - llama-3.3-70b-versatile [📊](frontier_model_test/llama-3.3-70b-versatile.png)
    - Error=\$98.24, RMSLE=0.70, Hits=44.8%
  - mistral-saba-24b [📊](frontier_model_test/mistral-saba-24b.png)
    - Error=\$98.02, RMSLE=0.82, Hits=44.8%
  - deepseek-r1-distill-llama-70b [📊](frontier_model_test/deepseek-r1-distill-llama-70b.png)
    - Error=\$109.09, RMSLE=0.67, Hits=48.4%
  - Human Evaluator [📊](frontier_model_test/human_pricer.png)
    - Error=\$126.55, RMSLE=1.00, Hits=32.0%
  - deepseek-r1-distill-qwen-32b [📊](frontier_model_test/deepseek-r1-distill-qwen-32b.png)
    - Error=\$151.59, RMSLE=0.80, Hits=38.4%

## frontier_model_tuning [🔗](frontier_model_tuning)
Fine-tuned gpt-4o-mini with the pickled training data from [data_curation](data_curation) using 200 samples.
- gpt-4o-mini-2024-07-18 [📊](frontier_model_tuning/gpt_fine_tuned.png)
    - Error=\$101.49, RMSLE=0.81, Hits=41.2%

## price_is_right_project [🔗](price_is_right_project)
Pulling together RAG, fine-tuned models, and adding agents. Using [Modal](https://modal.com/) to run it all.
- Load base and fine-tuned llama onto a Modal container as "pricer-service"
- Create a Chroma data store with 400,000 products
- Create a SpecialistAgent to call the pricing-service and return the result
- Create a FrontierAgent to call gpt-4o-mini and return the result
- Train a random forest model
- Create a RandomForestAgent to call the random forest model and return the result
- Train a linear regression model
- Create an EnsembleAgent that uses the Specialist, Frontier, and RandomForest agents to gather their prices. It then runs them against a linear regression model to provide its own price
- Create a deal scanner that scrapes RSS feed for deals and have gpt-4o-mini sort through them
- Create a MessagingAgent to send deal alerts via [Pushover](https://pushover.net/)
- Create a PlanningAgent that uses the other agents to create a process flow
- Create a Gradio UI on top of it all

## huggingface [🔗](huggingface)
- [01] GPT2 text generation
- [02] Playing with Tokenizers
- [03] Trying out different models for text classification, summarization, and generation
- [04] Getting, cleaning, and creating a custom dataset
- [05] Create a custom Tokenizer
- [06] Fine tuning
- [07] GPT2 fine tuning
- [08] Quantization
- [09] Multimodal
- [10] Natural Language Processing