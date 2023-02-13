# # Python program showing a use
# # of get() and set() method in
# # normal function

# class Geek:
# 	def __init__(self, age = 0):
# 		self._age = age
	
# 	# getter method
# 	def get_age(self):
# 		return self._age
	
# 	# setter method
# 	def set_age(self, x):
# 		self._age = x

# raj = Geek()

# # setting the age using setter
# raj.set_age(21)

# # retrieving age using getter
# print(raj.get_age())

# print(raj._age)

# Python program showing a
# use of property() function

# class Geeks:
# 	def __init__(self):
# 		self._age = 0
	
# 	# function to get value of _age
# 	def get_age(self):
# 		print("getter method called")
# 		return self._age
	
# 	# function to set value of _age
# 	def set_age(self, a):
# 		print("setter method called")
# 		self._age = a

# 	# function to delete _age attribute
# 	def del_age(self):
# 		del self._age
	
# 	age = property(get_age, set_age, del_age)

# mark = Geeks()


# mark.age = 10

# print(mark.age)


# Python program showing the use of
# @property

class Geeks:
	def __init__(self):
		self._age = 0
	
	# using property decorator
	# a getter function
	@property
	def age(self):
		print("getter method called")
		return self._age
	
	# a setter function
	@age.setter
	def age(self, a):
		if(a < 18):
			raise ValueError("Sorry you age is below eligibility criteria")
		print("setter method called")
		self._age = a

mark = Geeks()

mark.age = 19

print(mark.age)
