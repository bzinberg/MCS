import unittest

from mcs_util import MCS_Util

class My_Emptyclass:

    def __init__(self):
        pass

class My_Subclass:

    def __init__(self):
        self.my_integer = 7
        self.my_string = "h"
        self.my_list = [8, "i"]
        self.my_dict = {
            "my_integer": 9,
            "my_string": "j",
        }

    def __str__(self):
        return MCS_Util.class_to_str(self)

class My_Class:

    def __init__(self):
        self.my_boolean = True
        self.my_float = 1.234
        self.my_integer = 0
        self.my_string = "a"
        self.my_list = [1, "b", {
            "my_integer": 2,
            "my_string": "c",
            "my_list": [3, "d"]
        }]
        self.my_dict = {
            "my_integer": 4,
            "my_string": "e",
            "my_list": [5, "f"],
            "my_dict": {
                "my_integer": 6,
                "my_string": "g",
            }
        }
        self.my_list_empty = []
        self.my_dict_empty = {}
        self.my_subclass = My_Subclass()
        self.__my_private = "z"

    def my_function():
        pass

class Test_MCS_Util(unittest.TestCase):

    def test_class_to_str_with_class(self):
        self.maxDiff = 10000
        expected = "{\n    \"my_boolean\": True,\n    \"my_float\": 1.234,\n    \"my_integer\": 0,\n    \"my_string\": \"a\",\n    \"my_list\": [\n        1,\n        \"b\",\n        {\n            \"my_integer\": 2,\n            \"my_string\": \"c\",\n            \"my_list\": [\n                3,\n                \"d\"\n            ]\n        }\n    ],\n    \"my_dict\": {\n        \"my_integer\": 4,\n        \"my_string\": \"e\",\n        \"my_list\": [\n            5,\n            \"f\"\n        ],\n        \"my_dict\": {\n            \"my_integer\": 6,\n            \"my_string\": \"g\"\n        }\n    },\n    \"my_list_empty\": [],\n    \"my_dict_empty\": {},\n    \"my_subclass\": {\n        \"my_integer\": 7,\n        \"my_string\": \"h\",\n        \"my_list\": [\n            8,\n            \"i\"\n        ],\n        \"my_dict\": {\n            \"my_integer\": 9,\n            \"my_string\": \"j\"\n        }\n    }\n}"
        self.assertEqual(MCS_Util.class_to_str(My_Class()), expected)

    def test_class_to_str_with_empty_class(self):
        self.assertEqual(MCS_Util.class_to_str(My_Emptyclass()), "{}")

    def test_value_to_str_with_boolean(self):
        self.assertEqual(MCS_Util.value_to_str(True), "True")
        self.assertEqual(MCS_Util.value_to_str(False), "False")

    def test_value_to_str_with_dict(self):
        self.assertEqual(MCS_Util.value_to_str({}), "{}")
        self.assertEqual(MCS_Util.value_to_str({
            "number": 1,
            "string": "a"
        }), "{\n    \"number\": 1,\n    \"string\": \"a\"\n}")

    def test_value_to_str_with_float(self):
        self.assertEqual(MCS_Util.value_to_str(0.0), "0.0")
        self.assertEqual(MCS_Util.value_to_str(1234.5678), "1234.5678")

    def test_value_to_str_with_integer(self):
        self.assertEqual(MCS_Util.value_to_str(0), "0")
        self.assertEqual(MCS_Util.value_to_str(1234), "1234")

    def test_value_to_str_with_list(self):
        self.assertEqual(MCS_Util.value_to_str([]), "[]")
        self.assertEqual(MCS_Util.value_to_str([1, "a"]), "[\n    1,\n    \"a\"\n]")

    def test_value_to_str_with_string(self):
        self.assertEqual(MCS_Util.value_to_str(""), "\"\"")
        self.assertEqual(MCS_Util.value_to_str("a b c d"), "\"a b c d\"")

