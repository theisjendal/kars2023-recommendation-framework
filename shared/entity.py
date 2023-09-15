from typing import Union, List


class Entity:
    def __init__(self, index: int, name: str, recommendable: bool, original_id: str = None,
                 description: Union[str, List[str]] = None):
        """
        Create an entity
        :param index: unique index of the entity
        :param entity_type: type of entity, e.g. actor, genre, or category.
        :param recommendable: if the entity can be recommended to a user, e.g. a publisher is not recommended, but a book.
        :param name: name of the entity, such as movie or book title, name of a publisher.
        :param original_id: original id in the source data, optional.
        :param description: more detailed text description of the entity, can be empty.
        """
        self.original_id = original_id
        self.index = index
        self.recommendable = recommendable
        self.name = name
        self.description = description
