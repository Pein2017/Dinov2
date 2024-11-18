#!/bin/bash
### Please run on the host enviroment, not pods.


PROTECTED_LOGS_DIR="bbu_logs"


if [ "$1" == "lock" ]; then
    # Set the append-only attribute to prevent deletion, allowing addition and modification
    sudo chattr +a "$PROTECTED_LOGS_DIR"
    echo "Deletion protection enabled: Users cannot delete or remove contents, but can add or modify."
elif [ "$1" == "unlock" ]; then
    # Remove the append-only attribute
    sudo chattr -a "$PROTECTED_LOGS_DIR"
    echo "Deletion protection removed: Users can delete or remove contents."
else
    echo "Usage: $0 [lock|unlock]"
fi