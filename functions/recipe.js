exports.handler = async function(event) {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: JSON.stringify({ error: "Method not allowed" }) };
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return { statusCode: 500, body: JSON.stringify({ error: { message: "ANTHROPIC_API_KEY not set in environment" } }) };
  }

  let body;
  try {
    body = JSON.parse(event.body);
  } catch(e) {
    return { statusCode: 400, body: JSON.stringify({ error: { message: "Invalid JSON body" } }) };
  }

  // Build the messages array — handle both recipe and image scoring
  let messages;
  if (body.image) {
    messages = [{
      role: "user",
      content: [
        { type: "image", source: { type: "base64", media_type: body.mediaType || "image/jpeg", data: body.image } },
        { type: "text", text: `Score this dish called "${body.recipeTitle || "this dish"}". Be a tough but fair judge. Respond ONLY with JSON, nothing else: {"score":7,"verdict":"...","highlight":"...","improve":"..."}` }
      ]
    }];
  } else {
    messages = body.messages;
  }

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-5",
      max_tokens: body.max_tokens || 1200,
      messages: messages,
    }),
  });

  const data = await response.json();

  return {
    statusCode: response.status,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
};
