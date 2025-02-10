"""Util functions borrowed from TinyZero: https://github.com/Jiayi-Pan/TinyZero."""
import json
import re
from typing import Any, List, Tuple

import regex as re
import torch
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    ground_truth = json.loads(ground_truth)
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    do_print = True

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


class CountdownOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the Countdown math task.
    
    This oracle evaluates responses to countdown-style math problems where:
    - A target number must be reached
    - Using a specific set of numbers
    - With basic arithmetic operations
    - Responses are in XML format with <answer> tags
    
    Example:
        >>> oracle = CountdownOracle()
        >>> response = "Let's calculate... <answer>25 + 10</answer>"
        >>> reference = '{"target": 35, "numbers": [25, 10]}'
        >>> oracle.get_reward([], [response], [reference])
        (tensor([1.0]), {})
    """

    def __init__(self, format_score: float = 0.1, correct_score: float = 1.0) -> None:
        """Initialize the CountdownOracle.
        
        Args:
            format_score: Score for correct format but wrong answer (default: 0.1)
            correct_score: Score for correct answer (default: 1.0)
        """
        super().__init__()
        self.format_score = format_score
        self.correct_score = correct_score

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        """Calculate rewards for countdown responses.
        
        Args:
            inputs: List of input prompts (unused)
            responses: List of response strings to evaluate
            references: List of reference JSON strings containing target and numbers
            batch_size: Batch size for processing (unused)
            
        Returns:
            Tuple of (rewards tensor, metrics dictionary)
            
        Raises:
            ValueError: If responses and references have different lengths
        """
        if len(responses) != len(references):
            raise ValueError(f"Responses and references must be same length. Got {len(responses)} responses and {len(references)} references")

        rewards = []
        metrics = {
            'total': len(responses),
            'correct': 0,
            'format_correct': 0,
            'invalid': 0
        }

        for resp, ref in zip(responses, references):
            try:
                r = compute_score(resp, ref, self.format_score, self.correct_score)
                rewards.append(r)
                
                # Update metrics
                if r == self.correct_score:
                    metrics['correct'] += 1
                elif r == self.format_score:
                    metrics['format_correct'] += 1
                else:
                    metrics['invalid'] += 1
                    
            except Exception as e:
                print(f"Error evaluating response: {e}")
                rewards.append(0.0)
                metrics['invalid'] += 1

        return torch.tensor(rewards), metrics

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Compare two sets of candidate responses.
        
        Args:
            inputs: List of input prompts
            candidates_A: First set of candidate responses
            candidates_B: Second set of candidate responses
            batch_size: Batch size for processing
            return_probs: Whether to return probabilities
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            Tuple of (comparison results, metrics)
        """
        # Validate input lengths
        if len(candidates_A) != len(candidates_B):
            raise ValueError(f"Candidate sets must be same length. Got {len(candidates_A)} and {len(candidates_B)}")

        # Get rewards for both sets
        rewards_A, metrics_A = self.get_reward(inputs, candidates_A, candidates_B, batch_size)
        rewards_B, metrics_B = self.get_reward(inputs, candidates_B, candidates_A, batch_size)
        
        # Calculate comparison results
        comparison = (rewards_A > rewards_B).numpy().astype(int)
        
        # Combine metrics
        combined_metrics = {
            'total': metrics_A['total'],
            'correct_A': metrics_A['correct'],
            'correct_B': metrics_B['correct'],
            'prefer_A': int((rewards_A > rewards_B).sum()),
            'prefer_B': int((rewards_B > rewards_A).sum()),
            'tie': int((rewards_A == rewards_B).sum())
        }
        
        return comparison, combined_metrics
