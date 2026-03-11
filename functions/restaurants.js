// netlify/functions/restaurants.js
// Proxies requests to Yelp Fusion API — keeps your API key server-side

export default async (req) => {
  if (req.method !== "GET") {
    return new Response("Method not allowed", { status: 405 });
  }

  const url      = new URL(req.url);
  const params   = url.searchParams.toString();
  const yelpUrl  = `https://api.yelp.com/v3/businesses/search?${params}`;

  const response = await fetch(yelpUrl, {
    headers: {
      Authorization: `Bearer ${process.env.YELP_API_KEY}`,
      "Content-Type": "application/json",
    },
  });

  const data = await response.json();

  return new Response(JSON.stringify(data), {
    status: response.status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
  });
};

export const config = { path: "/api/restaurants" };
