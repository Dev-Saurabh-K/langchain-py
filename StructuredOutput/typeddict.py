from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name':'saurabh', 'age':135}

print(new_person)