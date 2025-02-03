# LLM Engineering
This project has use cases I used to learn more about LLM engineering.

Local models are on a VirtualBox VM with 16GB RAM and 8 CPUs.

## website_summarizer
- Use OpenAI API to summarize web page content and return the summary in Markdown.
- Do the same with a local version of llama3.2 and deepseek-r1:1.5b.
- Responses are saved by model name.

Local models spiked VM CPU 100% causing desktop fans to come on. Lots of resources for a seemingly simple task, although the URL used does produce around 12k characters.

## brochure_builder
- Use OpenAI API to get base web page content and links
- Send links via API asking to decide which links are relevant for a brochure
- Get page contents for each relevant link
- Send all page content back to API and build a brochure

This would be better implemented in something like CrewAI, but I wanted to do it the hard way first.