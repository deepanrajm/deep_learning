from mcp.server.fastmcp import FastMCP
import requests
import json

mcp = FastMCP("CryptoPriceDemo")

@mcp.tool()
def get_crypto_price(cryptocurrency: str, currency: str = "usd") -> str:
    """
    Fetches the current price of a given cryptocurrency in a specified currency.

    Args:
        cryptocurrency: The name of the cryptocurrency (e.g., 'bitcoin', 'ethereum').
        currency: The currency to get the price in (e.g., 'usd', 'inr'). Defaults to 'usd'.
    """
    # Sanitize inputs to be lowercase, as expected by the API
    crypto_id = cryptocurrency.lower()
    vs_currency = currency.lower()

    # Construct the API URL
    api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies={vs_currency}"
    
    try:
        # Make the GET request to the CoinGecko API
        response = requests.get(api_url)
        response.raise_for_status()  # Check for errors like 404 or 500

        # Parse the JSON response
        data = response.json()
        
        # Check if the API returned data for our requested cryptocurrency
        if crypto_id in data and vs_currency in data[crypto_id]:
            price = data[crypto_id][vs_currency]
            # Format the output string and return it
            return f"The current price of 1 {crypto_id.capitalize()} is {price:,.2f} {vs_currency.upper()}."
        else:
            return f"Could not find the price for '{cryptocurrency}'. Please check the name and try again."

    except requests.exceptions.HTTPError as http_err:
        return f"An HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as e:
        # Handle network-related errors
        return f"An error occurred while calling the API: {e}"
    except json.JSONDecodeError:
        return "Failed to decode the API response. The service may be down."

if __name__ == "__main__":
    mcp.run()