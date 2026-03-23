exports.handler = async function(event) {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method Not Allowed" };
  }

  const https = require("https");
  const body  = JSON.parse(event.body);
  const key   = process.env.ANTHROPIC_API_KEY;

  if (!key) {
    return { statusCode: 500, body: JSON.stringify({ error: { message: "No API key configured" } }) };
  }

  // Build the Anthropic request
  let anthropicBody;

  if (body.image) {
    // Photo scoring request
    anthropicBody = {
      model: "claude-sonnet-4-5",
      max_tokens: 600,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: body.mediaType, data: body.image } },
          { type: "text", text: `Score this dish called "${body.recipeTitle}". Be a tough but fair judge. Respond ONLY with JSON: {"score":7,"verdict":"...","highlight":"...","improve":"..."}` }
        ]
      }]
    };
  } else {
    // Recipe or restaurant request
    anthropicBody = {
      model: "claude-sonnet-4-5",
      max_tokens: body.max_tokens || 1200,
      messages: body.messages,
    };
  }

  const payload = JSON.stringify(anthropicBody);

  return new Promise((resolve) => {
    const req = https.request(
      {
        hostname: "api.anthropic.com",
        path: "/v1/messages",
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => { data += chunk; });
        res.on("end", () => {
          resolve({
            statusCode: res.statusCode,
            headers: { "Content-Type": "application/json" },
            body: data,
          });
        });
      }
    );

    req.on("error", (e) => {
      resolve({
        statusCode: 500,
        body: JSON.stringify({ error: { message: e.message } }),
      });
    });

    req.write(payload);
    req.end();
  });
};
