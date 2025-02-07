# LLM Engineering
I used these projects to learn more about LLM engineering. They're in completion order.

Local models are on a VirtualBox VM with 16GB RAM and 8 CPUs.

## website_summarizer
- Use OpenAI API to summarize web page content and return the summary in Markdown.
- Do the same with a local version of llama3.2 and deepseek-r1:1.5b.
- Responses are saved by model name.
    - [deepseek-r1:1.5b-response.md](website_summarizer/deepseek-r1:1.5b-response.md)
    - [gpt-4o-mini-response.md](website_summarizer/gpt-4o-mini-response.md)
    - [llama3.2-response.md](website_summarizer/llama3.2-response.md)

Local models spiked VM CPU 100% causing desktop fans to come on. Lots of resources for a seemingly simple task, although the URL used does produce around 12k characters.

## brochure_builder
- Use OpenAI API to get base web page content and links
- Send links via API asking to decide which links are relevant for a brochure
- Get page contents for each relevant link
- Send all page content back to API and build a brochure
- Output saved in [brochure.md](brochure_builder/brochure.md)

This would be better implemented in something like CrewAI, but I wanted to do it the hard way first.

## multi_model_conversation
- Created a 3-way conversation between models
    - gpt-4o-mini is an argumentative, snarky bot
    - claude-3-5-haiku is the polite bot
    - gemini-2.0-flash-exp is a couples counselor impersonating Sigmond Freud
- Output saved in [conversation.txt](multi_model_conversation/conversation.txt)

## gradio_llm_chat
- Basic GPT chat using Gradio

## basic_tool_call
- Built on **gradio_llm_chat** to mock a basic airline flight pricing chatbot.
- Handles "tool_calls"

I can now see why frameworks like CrewAI are popular - it's a pain to code this.

