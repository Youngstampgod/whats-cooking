const https = require("https");

exports.handler = async function(event) {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: JSON.stringify({ error: { message: "Method not allowed" } }) };
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return { statusCode: 500, body: JSON.stringify({ error: { message: "ANTHROPIC_API_KEY not set" } }) };
  }

  let body;
  try {
    body = JSON.parse(event.body);
  } catch(e) {
    return { statusCode: 400, body: JSON.stringify({ error: { message: "Invalid JSON" } }) };
  }

  let messages;
  if (body.snapMode && body.image) {
    messages = [{
      role: "user",
      content: [
        { type: "image", source: { type: "base64", media_type: body.mediaType || "image/jpeg", data: body.image } },
        { type: "text", text: `Look at this photo. Do NOT describe it literally. Instead, use its colors, textures, mood, energy, and aesthetic to inspire a food recommendation.\n\nWhat should this person eat tonight, based purely on the vibe of this photo?\n\nRespond ONLY with this JSON, nothing else:\n{"vibe_reading":"one evocative sentence about the energy/feeling of this photo","dish":"specific dish name (e.g. Spicy Crispy Honey Chicken, not just chicken)","why":"2 sentences explaining why this dish matches the photo's aesthetic and mood","cuisine":"cuisine type (e.g. Korean, Italian, Mexican)","time":"30 min"}` }
      ]
    }];
  } else if (body.image) {
    messages = [{
      role: "user",
      content: [
        { type: "image", source: { type: "base64", media_type: body.mediaType || "image/jpeg", data: body.image } },
        { type: "text", text: `Score this dish called "${body.recipeTitle || "this dish"}". Respond ONLY with JSON: {"score":7,"verdict":"...","highlight":"...","improve":"..."}` }
      ]
    }];
  } else {
    messages = body.messages;
  }

  const payload = JSON.stringify({
    model: "claude-sonnet-4-6",
    max_tokens: body.max_tokens || 1200,
    messages: messages,
  });

  return new Promise((resolve) => {
    const options = {
      hostname: "api.anthropic.com",
      path: "/v1/messages",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload),
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
    };

    const req = https.request(options, (res) => {
      let data = "";
      res.on("data", chunk => data += chunk);
      res.on("end", () => {
        resolve({
          statusCode: res.statusCode,
          headers: { "Content-Type": "application/json" },
          body: data,
        });
      });
    });

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
