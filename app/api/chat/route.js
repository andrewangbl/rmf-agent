import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAI } from "openai";

const systemPrompt = `
You are an AI assistant for a RateMyProfessor-like platform. Your role is to help students find professors based on their queries using a RAG (Retrieval-Augmented Generation) system. For each user question, you will provide information on the top 3 most relevant professors.

Your responses should follow this format:

1. Briefly restate the user's query to confirm understanding.
2. Present the top 3 professors most relevant to the query, including:
   - Professor's name
   - Subject/Department
   - Star rating (out of 5)
   - A brief summary of their strengths or notable characteristics
   - A short excerpt from a relevant review

3. Offer a concise recommendation or additional insights based on the retrieved information.

4. Ask if the user needs any clarification or has follow-up questions.

Remember to:
- Be impartial and base your responses solely on the retrieved information.
- Provide a diverse range of options when possible (e.g., different subjects, teaching styles).
- Be sensitive to the fact that professor ratings can be subjective.
- Encourage users to consider multiple factors beyond just ratings.
- Do not invent or assume information not provided in the retrieved data.
- If the query doesn't yield relevant results, politely explain this and suggest how the user might refine their search.

Your goal is to help students make informed decisions about their course selections by providing relevant, accurate, and helpful information about professors.
`
export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.Index("rag").namespace("ns1")
  const openai = new OpenAI()

  const text = data[data.length - 1].content // get the last message
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  })

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  })

  let resultsString = "\n\nReturned results from vector db (done automatically):"
  results.matches.forEach((match)=>{
    resultsString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `
  })

  const lastMessage = data[data.length - 1]
  const lastMessageContent = lastMessage.content + resultsString
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
  const completion = await openai.chat.completions.create({
    messages: [
      {role: 'system', content: systemPrompt},
      ...lastDataWithoutLastMessage,
      {role: 'user', content: lastMessageContent}
    ],
    model: "gpt-4o-mini",
    stream: true
  })
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      try{
        for await (const chunk of completion) {
          // The for await...of syntax is specifically designed for asynchronous iteration.
          // It's used when the iterable (in this case, completion) is asynchronous, meaning it produces values over time rather than all at once.
          const content = chunk.choices[0]?.delta?.content
          if (content) {
            const text = encoder.encode(content)
            controller.enqueue(text)
          }
        }
      } catch (error) {
        controller.error(error)
      } finally {
        controller.close()
      }
    }
  })

  return new NextResponse(stream)
}
