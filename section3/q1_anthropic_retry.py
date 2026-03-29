"""
Section 3 — Q1: Anthropic API Retry Function
Write a Python function that calls the Anthropic API and retries up to 3 times
on rate limit errors.
"""

import time
import anthropic


def call_anthropic_with_retry(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """
    Calls the Anthropic API with automatic retry on rate limit errors.

    Args:
        prompt: The user message to send.
        model: Anthropic model ID.
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Base delay in seconds, doubles each retry (exponential backoff).

    Returns:
        The assistant's response text.

    Raises:
        anthropic.RateLimitError: If still rate-limited after max_retries.
        anthropic.APIError: For non-rate-limit API errors (not retried).
    """
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env variable

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except anthropic.RateLimitError as e:
            if attempt == max_retries:
                print(f"Rate limit exceeded after {max_retries} retries.")
                raise  # Re-raise after final attempt

            wait_time = base_delay * (2 ** (attempt - 1))  # Exponential backoff
            print(f"[Attempt {attempt}/{max_retries}] Rate limited. Retrying in {wait_time}s...")
            time.sleep(wait_time)

        except anthropic.APIError as e:
            # Do NOT retry on non-rate-limit errors (auth, bad request, etc.)
            print(f"API error (not retried): {e}")
            raise


# --- Demo usage ---
if __name__ == "__main__":
    try:
        result = call_anthropic_with_retry("Say hello in 3 languages.")
        print(f"Response: {result}")
    except anthropic.RateLimitError:
        print("Failed: Rate limit exceeded after all retries.")
    except anthropic.APIError as e:
        print(f"Failed: {e}")
