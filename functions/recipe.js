exports.handler = async (event) => {
  const body = JSON.parse(event.body);

  // Image scoring request
  if (body.image) {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": process.env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 500,
        messages: [{
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: body.mediaType,
                data: body.image,
              },
            },
            {
              type: "text",
              text: `You are a brutally honest but fair culinary judge with Michelin-level standards and Anthony Bourdain's directness. 

The cook was given this challenge: ${body.recipeTitle}

Score their finished dish from 1-10 based on what you see. Be specific about what you observe — plating, color, texture, apparent doneness, portioning. Don't be cruel but don't coddle.

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "score": 8,
  "verdict": "Two to three sentence judge's verdict. Specific, direct, honest.",
  "highlight": "The single best thing about this plate",
  "improve": "The single most important thing to do better next time"
}`
            }
          ]
        }]
      }),
    });
    const data = await response.json();
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    };
  }

  // Standard recipe request
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
};
