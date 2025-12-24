In this project, you are going to make a chatbot to scrape LLM Inference Serving websites to research costs of serving various LLMs. You will do this by writing an MCP Server that hooks up to Firecrawl's API and saving the data in a SQLite Database. You should use the following websites to scrape:

- "cloudrift": "https://www.cloudrift.ai/inference"
- "deepinfra": "https://deepinfra.com/pricing"
- "fireworks": "https://fireworks.ai/pricing#serverless-pricing"
- "groq": "https://groq.com/pricing"

4. Complete the 2 tool calls in `server.py`
5. Complete any section in `client.py` that has "#complete".
6. Test using any methods taught in the course
7. Use the following prompts in your chatbot but play around with all the LLM providers in the list above:
   - "How much does cloudrift ai (https://www.cloudrift.ai/inference) charge for deepseek v3?"
   - "How much does deepinfra (https://deepinfra.com/pricing) charge for deepseek v3"
   - "Compare cloudrift ai and deepinfra's costs for deepseek v3"

first, scrape these sites: {"cloudrift": "https://www.cloudrift.ai/inference", "deepinfra": "https://deepinfra.com/pricing", "fireworks": "https://fireworks.ai/pricing#serverless-pricing", "groq": "https://groq.com/pricing"}