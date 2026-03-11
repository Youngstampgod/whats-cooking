exports.handler = async (event) => {
  const params = new URLSearchParams(event.queryStringParameters).toString();
  const response = await fetch(`https://api.yelp.com/v3/businesses/search?${params}`, {
    headers: {
      Authorization: `Bearer ${process.env.YELP_API_KEY}`,
    },
  });
  const data = await response.json();
  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
};
