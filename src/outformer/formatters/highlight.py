from termcolor import colored
from typing import Iterable, Union


def highlight_values(
    values: Union[dict, list, str],
    color: Union[str, tuple[int, int, int], None] = "magenta",
    on_color: Union[str, tuple[int, int, int], None] = None,
    attrs: Union[Iterable[str], None] = None,
) -> None:
    """
    Recursively prints a JSON object with highlighted values.

    Args:
        values (Union[dict, list, str]): The JSON object to print
        color (Union[str, tuple[int, int, int], None]): The color to use for the highlighted values
        on_color (Union[str, tuple[int, int, int], None]): The color to use for the background of the highlighted values
        attrs (Union[Iterable[str], None]): Additional attributes to use for the highlighted values
    """

    def recursive_print(
        obj: Union[dict, list, str], indent: int = 0, is_last_element: bool = True
    ) -> None:
        """
        Helper function that recursively prints the JSON structure.

        Args:
            obj (Union[dict, list, str]): The current object to print
            indent (int): The current indentation level
            is_last_element (bool): Whether this is the last element in an array/object
        """
        if isinstance(obj, dict):
            # Print opening brace for objects
            print("{")

            # Get the last key to determine when to add commas
            last_key = list(obj.keys())[-1] if obj else None

            # Print each key-value pair
            for key, value in obj.items():
                print(f"{' ' * (indent + 2)}{key}: ", end="")
                recursive_print(
                    value, indent=indent + 2, is_last_element=key == last_key
                )

            # Print closing brace with appropriate comma
            print(f"{' ' * (indent)}}}", end="\n" if is_last_element else ",\n")

        elif isinstance(obj, list):
            # Print opening bracket for arrays
            print("[")
            array_length = len(obj)

            # Print each element in the array
            for index, element in enumerate(obj):
                print(f"{' ' * (indent + 2)}", end="")
                recursive_print(
                    obj=element,
                    indent=indent + 2,
                    is_last_element=index == array_length - 1,
                )

            # Print closing bracket with appropriate comma
            print(f"{' ' * (indent)}]", end="\n" if is_last_element else ",\n")

        else:
            # For primitive values (strings, numbers, booleans)
            if isinstance(obj, str):
                obj = f'"{obj}"'

            # Highlight the value in green
            print(
                colored(text=obj, color=color, on_color=on_color, attrs=attrs),
                end="\n" if is_last_element else ",\n",
            )

    # Start the recursive print
    recursive_print(obj=values)
