def read_text_file(file_path):
    """
    Reads the content of a text file and returns it as a string.

    Args:
        file_path (str): The path to the text file to be read.

    Returns:
        str: The contents of the file if successful, or an error message if the file cannot be found or an error occurs.

    Raises:
        FileNotFoundError: If the file is not found at the given path.
        Exception: For any other unexpected errors.
    """

    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"
