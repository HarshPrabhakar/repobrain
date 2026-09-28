def calculate_tax(amount):
    """Calculate the sales tax on an amount."""
    return amount * 0.18


def checkout(amount):
    """Calculate the total checkout price including tax."""
    return amount + calculate_tax(amount)


def refund(amount):
    """Refund an amount without applying sales tax."""
    return -amount
