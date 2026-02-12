import re
from openai import OpenAI


def extract_number_from_response(response_text):
    """
    Extract the last number found in the AI response text.
    This handles cases where the AI model includes the count in different formats.
    """
    # Find all numbers in the response
    numbers = re.findall(r'\d+', response_text)
    
    if numbers:
        # Return the last number found (usually the total count)
        return numbers[-1]
    else:
        return "0"

def detect_vehicles_from_image(image_url):

  client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = "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",
  )

  completion = client.chat.completions.create(
    extra_body = {},
    model = "allenai/molmo-2-8b",
    messages = [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Count the number of vehicles in this image"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": image_url
            }
          },
        ]
      }
    ]
  )

  # Extract just the number from the AI response
  ai_response = completion.choices[0].message.content
  vehicle_count = extract_number_from_response(ai_response)

  # Print only the number as requested
  print(vehicle_count)
  return ai_response, vehicle_count


if __name__ == "__main__":
  detect_vehicles_from_image("https://cameras.qldtraffic.qld.gov.au/Metropolitan/MRMETRO-1213.jpg?1770359141831")