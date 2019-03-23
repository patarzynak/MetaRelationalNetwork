import random
from enum import Enum
import numpy
import cairo
import math
from functools import partial
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from constraint import *
import random
import timeit
import itertools
import torch
from torch.utils.data.dataset import Dataset



#region Utility functions

def bgr01(r, g, b):
    """Rescales BGR to a 0-1 scale."""
    return (b / 255.0, g / 255.0, r / 255.0)

#endregion

#region Colours

class ConcreteColour(object):
    def __init__(self, name, r, g, b):
        self.name = name
        self.r = r
        self.g = g
        self.b = b
    
    def to_bgr01(self):
        return (self.b / 255.0, self.g / 255.0, self.r / 255.0)

#endregion

#region Shapes

class ConcreteShape(object):
    """Draw the shape on a Cairo context.
    
    Arguments:
    cr -- Cairo context
    position -- (x, y) position of the center of the shape.
    size -- size of the bounding box; the shape should be entirely contained within
            a square with sides of length SIZE centered at POSITION.
    colour -- the colour with which to fill the shape."""
    @classmethod
    def draw(cr, position, size, colour):
        raise NotImplementedError

    @classmethod
    def to_tikz(self, position, size, colour):
        pass

class Square(ConcreteShape):
    name = "square"

    @classmethod
    def draw(self, cr, position, size, colour):
        x, y = position
        cr.rectangle(x - size / 2, y - size / 2, size, size)
        cr.set_source_rgb(*colour.to_bgr01())
        cr.fill_preserve()
        cr.set_line_width(1)
        cr.set_source_rgb(*bgr01(33, 33, 33))
        cr.stroke()

    @classmethod
    def to_tikz(self, position, size, colour):
        x, y = position
        tikz_str = ""
        tikz_str += f"\\fill [{colour.name}] ({x - size / 2}, {y - size / 2}) rectangle ({x + size / 2}, {y + size / 2});\n"
        tikz_str += f"\\draw ({x - size / 2}, {y - size / 2}) rectangle ({x + size / 2}, {y + size / 2});\n"
        return tikz_str

class Circle(ConcreteShape):
    name = "circle"

    @classmethod
    def draw(self, cr, position, size, colour):
        x, y = position
        cr.arc(x, y, size / 2, 0, 2*math.pi)
        cr.set_source_rgb(*colour.to_bgr01())
        cr.fill_preserve()
        cr.set_line_width(1)
        cr.set_source_rgb(*bgr01(33, 33, 33))
        cr.stroke()
    
    @classmethod
    def to_tikz(self, position, size, colour):
        x, y = position
        tikz_str = ""
        tikz_str += f"\\fill [{colour.name}] ({x}, {y}) circle ({size / 2});\n"
        tikz_str += f"\\draw ({x}, {y}) circle ({size / 2});\n"
        return tikz_str

class Triangle(ConcreteShape):
    name = "triangle"
    
    @classmethod
    def draw(self, cr, position, size, colour):
        x, y = position
        cr.move_to(x - size / 2, y + size / 2)
        cr.line_to(x, y + size / 2 - math.sqrt(size ** 2 - (size / 2) ** 2))
        cr.line_to(x + size / 2, y + size / 2)
        cr.line_to(x - size / 2, y + size / 2)
        cr.set_source_rgb(*colour.to_bgr01())
        cr.fill_preserve()
        cr.set_line_width(1)
        cr.set_source_rgb(*bgr01(33, 33, 33))
        cr.stroke()

    @classmethod
    def to_tikz(self, position, size, colour):
        x, y = position
        tikz_str = ""        
        tikz_str += f"\\fill [{colour.name}] ({x - size / 2}, {y + size / 2}) -- ({x}, {y + size / 2 - math.sqrt(size ** 2 - (size / 2) ** 2)}) -- ({x + size / 2}, {y + size / 2}) -- cycle;\n"
        tikz_str += f"\\draw ({x - size / 2}, {y + size / 2}) -- ({x}, {y + size / 2 - math.sqrt(size ** 2 - (size / 2) ** 2)}) -- ({x + size / 2}, {y + size / 2}) -- cycle;\n"
        return tikz_str


#endregion

class AnswerType(Enum):
    COLOUR = 0
    SHAPE = 1
    COUNT = 2
    LEFT_OR_RIGHT = 3
    ABOVE_OR_BELOW = 4
    CLOSEST_OR_FURTHEST = 5
    YES_OR_NO = 6
    TOP_OR_BOTTOM = 7 # comment this out ???

class Relation(Enum):
    FURTHEST = 0
    CLOSEST = 1
    LEFT = 2
    RIGHT = 3
    ABOVE = 4
    BELOW = 5
    COLOUR = 6
    SHAPE = 7
    COUNT = 8

    def __str__(self):
        if self == Relation.FURTHEST:
            return "furthest from"
        elif self == Relation.CLOSEST:
            return "closest to"
        elif self == Relation.LEFT:
            return "on the left of"
        elif self == Relation.RIGHT:
            return "on the right of"
        elif self == Relation.ABOVE:
            return "above"
        elif self == Relation.BELOW:
            return "below"
        elif self == Relation.COLOUR:
            return "the same colour as"
        elif self == Relation.SHAPE:
            return "the same shape as"
        elif self == Relation.COUNT:
            return "count relation"

class Aspect(Enum):
    COLOUR = 0
    SHAPE = 1

class Shape(Enum):
    SHAPE_0 = 0
    SHAPE_1 = 1
    SHAPE_2 = 2

    @classmethod
    def random(self, excluding=None):
        shapes = [self.SHAPE_0, self.SHAPE_1, self.SHAPE_2]
        return random.choice([shape for shape in shapes if shape != excluding])

class Colour(Enum):
    COLOUR_0 = 0
    COLOUR_1 = 1
    COLOUR_2 = 2

    @classmethod
    def random(self, excluding=None):
        colours = [self.COLOUR_0, self.COLOUR_1, self.COLOUR_2]
        return random.choice([colour for colour in colours if colour != excluding])

def Count(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2

    @classmethod
    def random(self):
        return random.choice([self.ZERO, self.ONE, self.TWO])

class Point(object):
    def __init__(self, x, y):
        self.y = y
        self.x = x
    
    def distance_to(self, point):
        return abs(self.x - point.x) + abs(self.y - point.y)
        #return math.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def is_left_of(self, point):
        return self.x < point.x
    
    def is_right_of(self, point):
        return self.x > point.x

    def is_above(self, point):
        return self.y < point.y

    def is_below(self, point):
        return self.y > point.y

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

class Config(object):
    def __init__(self, concrete_shapes, concrete_colours, grid_size, grid_step, shape_size):
        self.concrete_shapes = concrete_shapes
        self.concrete_colours = concrete_colours
        self.grid_size = grid_size
        self.grid_step = grid_step
        self.shape_size = shape_size

class Scene(object):
    def __init__(self, config, placed_objects):
        self.config = config
        self.placed_objects = placed_objects

    def to_numpy(self):
        # Set up a numpy array which will hold a sccene (an image), encoded in RGBA.
        data = numpy.zeros((self.config.grid_size, self.config.grid_size, 4), dtype=numpy.uint8)
        # Create a Cairo surface to draw shapes on. It will use the numpy array created above as backing storage.
        surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_RGB24, self.config.grid_size, self.config.grid_size)
        # Create a Cairo context to draw on
        cr = cairo.Context(surface)
        # Fill the scene with a white background
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()

        for (shape, colour, position) in self.placed_objects:
            self.config.concrete_shapes[shape].draw(cr, (position.x, position.y), self.config.shape_size, self.config.concrete_colours[colour])

        return data[:, :, :3]
    
    def to_tensor(self):
        return torch.from_numpy(self.to_numpy())

    def to_tikz(self):
        output = ""
        output += "\\begin{tikzpicture}[y=-1cm]"
        output += "\\begin{axis}[axis equal image,xmin=0,xmax=48,ymin=0,ymax=48,yscale=-1]"
        for (shape, colour, position) in self.placed_objects:
            output += self.config.concrete_shapes[shape].to_tikz((position.x, position.y), self.config.shape_size, self.config.concrete_colours[colour])
        output += "\\end{axis}"
        output += "\\end{tikzpicture}"
        return output

    def show(self):
        plt.imshow(self.to_numpy())

class SceneBuilder(object):
    def __init__(self, config):
        self.config = config
        self.problem = Problem() #MinConflictsSolver()

        shapes = [Shape.SHAPE_0, Shape.SHAPE_1, Shape.SHAPE_2]
        colours = [Colour.COLOUR_0, Colour.COLOUR_1, Colour.COLOUR_2]
        positions = [Point(x, y) for x in range(config.grid_step, config.grid_size, config.grid_step)
                                                                   for y in range(config.grid_step, config.grid_size, config.grid_step)]
        for i in range(5):
            self.problem.addVariable((i, "shape"), random.sample(shapes, len(shapes)))
            self.problem.addVariable((i, "colour"), random.sample(colours, len(colours)))
            self.problem.addVariable((i, "position"), random.sample(positions, len(positions)))

            for j in range(5):
                if i != j:
                    self.problem.addConstraint(lambda x, y: x != y, [(i, "position"), (j, "position")])

    def set_shape(self, index, shape, unique=False):
        self.problem.addConstraint(lambda x: x == shape, [(index, "shape")])
        if unique:
            for i in range(5):
                if i != index:
                    self.problem.addConstraint(lambda x: x != shape, [(i, "shape")])

    def set_colour(self, index, colour, unique=False):
        self.problem.addConstraint(lambda x: x == colour, [(index, "colour")])
        if unique:
            for i in range(5):
                if i != index:
                    self.problem.addConstraint(lambda x: x != colour, [(i, "colour")])

    def set_aspect(self, index, aspect, shape, colour, unique=False):
        if aspect == Aspect.COLOUR:
            self.set_colour(index, colour, unique)
        else:
            self.set_shape(index, shape, unique)

    def set_furthest_from(self, i, j):
        for k in range(2, 5):
            self.problem.addConstraint(
                lambda x, y, z: x.distance_to(z) < x.distance_to(y) - self.config.grid_size * 0.2,
                [(i, "position"), (j, "position"), (k, "position")],
            )

    def set_closest_to(self, i, j):
        for k in range(2, 5):
            self.problem.addConstraint(
                lambda x, y, z: x.distance_to(z) > x.distance_to(y) + self.config.grid_size * 0.2,
                [(i, "position"), (j, "position"), (k, "position")],
            )

    def set_left_of(self, i, j):
        self.problem.addConstraint(
            lambda a, b: a.is_left_of(b),
            [(i, "position"), (j, "position")],
        )

    def set_right_of(self, i, j):
        self.problem.addConstraint(
            lambda a, b: a.is_right_of(b),
            [(i, "position"), (j, "position")],
        )

    def set_left_of(self, i, j):
        self.problem.addConstraint(
            lambda a, b: a.is_left_of(b),
            [(i, "position"), (j, "position")],
        )

    def set_above(self, i, j):
        self.problem.addConstraint(
            lambda a, b: a.is_above(b),
            [(i, "position"), (j, "position")],
        )
    
    def set_left_of(self, i, j):
        self.problem.addConstraint(
            lambda a, b: a.is_left_of(b),
            [(i, "position"), (j, "position")],
        )

    def set_below(self, i, j):
        self.problem.addConstraint(
            lambda a, b: a.is_below(b),
            [(i, "position"), (j, "position")],
        )

    

    def build(self):    
        solution = self.problem.getSolution()
        placed_objects = []
        for i in range(5):
            shape = solution[(i, "shape")]
            colour = solution[(i, "colour")]
            position = solution[(i, "position")]
            placed_object = (shape, colour, position)
            placed_objects.append(placed_object)

        return Scene(self.config, placed_objects)

class Question(object):
    def primary_label(self):
        if self.primary_aspect == Aspect.COLOUR:
            return "{} object".format(self.config.concrete_colours[self.primary_colour].name)
        else:
            return self.config.concrete_shapes[self.primary_shape].name
        
    def secondary_label(self):
        if self.secondary_aspect == Aspect.COLOUR:
            return "{} object".format(self.config.concrete_colours[self.secondary_colour].name)
        else:
            return self.config.concrete_shapes[self.secondary_shape].name

    def to_numpy(self):
        # Encode answer type
        answer_type_vector = numpy.zeros(8)
        answer_type_vector[self.answer_type.value] = 1

        # Encode relation
        relation_vector = numpy.zeros(9)
        if self.relation is not None:
            relation_vector[self.relation.value] = 1

        # Encode primary object
        # - Encode primary aspect
        primary_aspect_vector = numpy.zeros(2)
        if self.primary_aspect is not None:
            primary_aspect_vector[self.primary_aspect.value] = 1
        # - Encode primary colour
        primary_colour_vector = numpy.zeros(3)
        if self.primary_colour is not None:
            primary_colour_vector[self.primary_colour.value] = 1
        # - Encode primary shape
        primary_shape_vector = numpy.zeros(3)
        if self.primary_shape is not None:
            primary_shape_vector[self.primary_shape.value] = 1

        primary_vector = numpy.concatenate((primary_aspect_vector, primary_colour_vector, primary_shape_vector))

        # Encode secondary object
        # - Encode secondary aspect
        secondary_aspect_vector = numpy.zeros(2)
        if self.secondary_aspect is not None:
            secondary_aspect_vector[self.secondary_aspect.value] = 1
        # - Encode secondary colour
        secondary_colour_vector = numpy.zeros(3)
        if self.secondary_colour is not None:
            secondary_colour_vector[self.secondary_colour.value] = 1
        # - Encode secondary shape
        secondary_shape_vector = numpy.zeros(3)
        if self.secondary_shape is not None:
            secondary_shape_vector[self.secondary_shape.value] = 1

        secondary_vector = numpy.concatenate((secondary_aspect_vector, secondary_colour_vector, secondary_shape_vector))

        return numpy.concatenate((answer_type_vector, relation_vector, primary_vector, secondary_vector))

    def to_tensor(self):
        return torch.from_numpy(self.to_numpy())

class Question1(Question):
    def __init__(self, config, answer_type, relation, primary_aspect):
        self.config = config
        self.answer_type = answer_type #random.choice([AnswerType.COLOUR, AnswerType.SHAPE])
        self.relation = relation #random.choice([Relation.FURTHEST, Relation.CLOSEST])
        self.primary_aspect = primary_aspect #random.choice([Aspect.COLOUR, Aspect.SHAPE])
        if self.primary_aspect == Aspect.COLOUR:
            self.primary_colour = Colour.random()
            self.primary_shape = None
        else:
            self.primary_colour = None
            self.primary_shape = Shape.random()
        self.secondary_aspect = None
        self.secondary_colour = None
        self.secondary_shape = None

    def __str__(self):
        question = "What is the {}".format("colour" if self.answer_type == AnswerType.COLOUR else "shape")
        question += " of the object {}".format("furthest away from" if self.relation == Relation.FURTHEST else "closest to")
        question += " the {}?".format(self.config.concrete_shapes[self.primary_shape].name if self.primary_aspect == Aspect.SHAPE else "{} object".format(self.config.concrete_colours[self.primary_colour].name))
        return question

    def get_random_answer(self):
        if self.answer_type == AnswerType.SHAPE:
            return Shape.random(excluding=self.primary_shape)
        elif self.answer_type == AnswerType.COLOUR:
            return Colour.random(excluding=self.primary_colour)

    def get_all_possible_answers(self):
        if self.answer_type == AnswerType.SHAPE:
            shapes = [Shape.SHAPE_0, Shape.SHAPE_1, Shape.SHAPE_2]
            return [x for x in shapes if x != self.primary_shape]
        elif self.answer_type == AnswerType.COLOUR:
            colours = [Colour.COLOUR_0, Colour.COLOUR_1, Colour.COLOUR_2]
            return [x for x in colours if x != self.primary_colour]

    def get_random_scene(self, answer):
        scene_builder = SceneBuilder(self.config)

        scene_builder.set_aspect(0, self.primary_aspect, self.primary_shape, self.primary_colour, unique=True)

        if self.answer_type == AnswerType.SHAPE:
            scene_builder.set_shape(1, answer)
        elif self.answer_type == AnswerType.COLOUR:
            scene_builder.set_colour(1, answer)

        if self.relation == Relation.FURTHEST:
            scene_builder.set_furthest_from(0, 1)
        else:
            scene_builder.set_closest_to(0, 1)

        scene = scene_builder.build()

        return scene

class Question2(Question):
    def __init__(self, config, answer_type, relation):
        self.config = config
        self.answer_type = answer_type #random.choice([AnswerType.COLOUR, AnswerType.SHAPE])
        self.relation = relation #random.choice([Relation.LEFT, Relation.RIGHT, Relation.ABOVE, Relation.BELOW])
        if self.answer_type == AnswerType.COLOUR:
            self.primary_aspect = Aspect.COLOUR
            self.primary_colour = Colour.random()
            self.primary_shape = None
        else:
            self.primary_aspect = Aspect.SHAPE
            self.primary_colour = None
            self.primary_shape = Shape.random()
            
        if self.answer_type == AnswerType.COLOUR:
            self.secondary_aspect = Aspect.SHAPE
            self.secondary_colour = None
            self.secondary_shape = Shape.random()
        else:
            self.secondary_aspect = Aspect.COLOUR
            self.secondary_colour = Colour.random()
            self.secondary_shape = None


    def __str__(self):
        question = "What is the {}".format("colour" if self.answer_type == AnswerType.COLOUR else "shape")
        question += " of the {}".format(self.secondary_label())
        question += " {}".format(self.relation)
        question += " the {}?".format(self.primary_label())
        return question

    def get_random_answer(self):
        if self.answer_type == AnswerType.SHAPE:
            return Shape.random(excluding=self.primary_shape)
        elif self.answer_type == AnswerType.COLOUR:
            return Colour.random(excluding=self.primary_colour)

    def get_all_possible_answers(self):
        if self.answer_type == AnswerType.SHAPE:
            shapes = [Shape.SHAPE_0, Shape.SHAPE_1, Shape.SHAPE_2]
            return [x for x in shapes if x != self.primary_shape]
        elif self.answer_type == AnswerType.COLOUR:
            colours = [Colour.COLOUR_0, Colour.COLOUR_1, Colour.COLOUR_2]
            return [x for x in colours if x != self.primary_colour]

    def get_random_scene(self, answer):
        scene_builder = SceneBuilder(self.config)

        scene_builder.set_aspect(0, self.primary_aspect, self.primary_shape, self.primary_colour, unique=True)
        scene_builder.set_aspect(1, self.secondary_aspect, self.secondary_shape, self.secondary_colour)

        if self.answer_type == AnswerType.SHAPE:
            scene_builder.set_shape(1, answer)
        elif self.answer_type == AnswerType.COLOUR:
            scene_builder.set_colour(1, answer)

        p = None
        if self.relation == Relation.LEFT:
            scene_builder.set_left_of(1, 0)
            p = lambda x, y: y.is_left_of(x)
        elif self.relation == Relation.RIGHT:
            scene_builder.set_right_of(1, 0)
            p = lambda x, y: y.is_right_of(x)
        elif self.relation == Relation.ABOVE:
            scene_builder.set_above(1, 0)
            p = lambda x, y: y.is_above(x)
        elif self.relation == Relation.BELOW:
            scene_builder.set_below(1, 0)
            p = lambda x, y: y.is_below(x)

        for i in range(2, 5):
            scene_builder.problem.addConstraint(
                lambda x, y, z:
                    not (p(x,y) and z == (self.secondary_shape if self.secondary_aspect == Aspect.SHAPE else self.secondary_colour)),
                [(0, "position"), (i, "position"), (i, "shape" if self.secondary_aspect == Aspect.SHAPE else "colour")]
            )

        # Add another
        scene_builder.problem.addConstraint(
                lambda x, y, z:
                    not p(x,y) and z == (self.secondary_shape if self.secondary_aspect == Aspect.SHAPE else self.secondary_colour),
                [(0, "position"), (4, "position"), (4, "shape" if self.secondary_aspect == Aspect.SHAPE else "colour")]
            )

        scene = scene_builder.build()

        return scene

class Question3(Question):
    def __init__(self, config, answer_type, primary_aspect, secondary_aspect):
        self.config = config
        self.answer_type = answer_type #random.choice([AnswerType.LEFT_OR_RIGHT, AnswerType.ABOVE_OR_BELOW, AnswerType.CLOSEST_OR_FURTHEST])
        self.relation = None
        self.primary_aspect = primary_aspect #random.choice([Aspect.SHAPE, Aspect.COLOUR])
        if self.primary_aspect == Aspect.SHAPE:
            self.primary_colour = None
            self.primary_shape = Shape.random()
        else:
            self.primary_colour = Colour.random()
            self.primary_shape = None
        self.secondary_aspect = secondary_aspect #random.choice([Aspect.SHAPE, Aspect.COLOUR])
        if self.secondary_aspect == Aspect.SHAPE:
            self.secondary_colour = None
            self.secondary_shape = Shape.random(excluding=self.primary_shape)
        else:
            self.secondary_colour = Colour.random(excluding=self.primary_colour)
            self.secondary_shape = None


    def __str__(self):
        question = "Is the {}".format(self.secondary_label())
        if self.answer_type == AnswerType.LEFT_OR_RIGHT:
            question += " on the left or on the right of"
        elif self.answer_type == AnswerType.ABOVE_OR_BELOW:
            question += " above or below"
        elif self.answer_type == AnswerType.CLOSEST_OR_FURTHEST:
            question += " closest to or furthest from"
        question += " {}?".format(self.primary_label())
        return question

    def get_random_answer(self):
        if self.answer_type == AnswerType.LEFT_OR_RIGHT:
            return random.choice([Relation.LEFT, Relation.RIGHT])
        elif self.answer_type == AnswerType.ABOVE_OR_BELOW:
            return random.choice([Relation.ABOVE, Relation.BELOW])
        elif self.answer_type == AnswerType.CLOSEST_OR_FURTHEST:
            return random.choice([Relation.CLOSEST, Relation.FURTHEST])

    def get_all_possible_answers(self):
        if self.answer_type == AnswerType.LEFT_OR_RIGHT:
            return [Relation.LEFT, Relation.RIGHT]
        elif self.answer_type == AnswerType.ABOVE_OR_BELOW:
            return [Relation.ABOVE, Relation.BELOW]
        elif self.answer_type == AnswerType.CLOSEST_OR_FURTHEST:
            return [Relation.CLOSEST, Relation.FURTHEST]

    def get_random_scene(self, answer):
        scene_builder = SceneBuilder(self.config)

        scene_builder.set_aspect(0, self.primary_aspect, self.primary_shape, self.primary_colour, unique=True)
        scene_builder.set_aspect(1, self.secondary_aspect, self.secondary_shape, self.secondary_colour, unique=True)

        if answer == Relation.LEFT:
            scene_builder.set_left_of(1, 0)
        elif answer == Relation.RIGHT:
            scene_builder.set_right_of(1, 0)
        elif answer == Relation.ABOVE:
            scene_builder.set_above(1, 0)
        elif answer == Relation.BELOW:
            scene_builder.set_below(1, 0)
        elif answer == Relation.FURTHEST:
            scene_builder.set_furthest_from(0, 1)
        elif answer == Relation.CLOSEST:
            scene_builder.set_closest_to(0, 1)

        scene = scene_builder.build()

        return scene

class Question4(Question):
    def __init__(self, config, relation):
        self.config = config
        self.answer_type = AnswerType.COUNT
        self.relation = relation #random.choice([Relation.SHAPE, Relation.COLOUR])
        if self.relation == Relation.SHAPE:
            self.primary_aspect = Aspect.COLOUR
        elif self.relation == Relation.COLOUR:
            self.primary_aspect = Aspect.SHAPE
        if self.primary_aspect == Aspect.SHAPE:
            self.primary_colour = None
            self.primary_shape = Shape.random()
        else:
            self.primary_colour = Colour.random()
            self.primary_shape = None
        self.secondary_aspect = None
        self.secondary_colour = None
        self.secondary_shape = None


    def __str__(self):
        question = "How many objects are {}".format(self.relation)
        question += " {}?".format(self.primary_label())
        return question

    def get_random_answer(self):
        return random.choice([1, 2, 3])

    def get_all_possible_answers(self):
        return [1, 2, 3]

    def get_random_scene(self, answer):
        scene_builder = SceneBuilder(self.config)

        scene_builder.set_aspect(0, self.primary_aspect, self.primary_shape, self.primary_colour, unique=True)

        scene_builder.problem.addConstraint(
            lambda x, *ys: sum([x == y for y in ys]) == answer,
            [(i, "shape" if self.relation == Relation.SHAPE else "colour") for i in range(0, 5)]
        )

        scene = scene_builder.build()
        
        return scene

class Question5(Question):
    def __init__(self, config, relation, primary_aspect):
        self.config = config
        self.answer_type = AnswerType.COUNT
        self.relation = relation #random.choice([Relation.LEFT, Relation.RIGHT, Relation.ABOVE, Relation.BELOW])
        self.primary_aspect = primary_aspect #random.choice([Aspect.SHAPE, Aspect.COLOUR])
        if self.primary_aspect == Aspect.SHAPE:
            self.primary_colour = None
            self.primary_shape = Shape.random()
        else:
            self.primary_colour = Colour.random()
            self.primary_shape = None
        self.secondary_aspect = None
        self.secondary_colour = None
        self.secondary_shape = None


    def __str__(self):
        question = "How many objects are {}".format(self.relation)
        question += " {}?".format(self.primary_label())
        return question

    def get_random_answer(self):
        return random.choice([1, 2, 3])

    def get_all_possible_answers(self):
        return [1, 2, 3]

    def get_random_scene(self, answer):
        scene_builder = SceneBuilder(self.config)

        scene_builder.set_aspect(0, self.primary_aspect, self.primary_shape, self.primary_colour, unique=True)

        p = None
        if self.relation == Relation.LEFT:
            p = lambda x, y: y.is_left_of(x)
        elif self.relation == Relation.RIGHT:
            p = lambda x, y: y.is_right_of(x)
        elif self.relation == Relation.ABOVE:
            p = lambda x, y: y.is_above(x)
        elif self.relation == Relation.BELOW:
            p = lambda x, y: y.is_below(x)
        
        for i in range(1, answer+1):
            scene_builder.problem.addConstraint(
                lambda x, y: p(x, y),
                [(0, "position"), (i, "position")]
            )

        for i in range(answer+1, 5):
            scene_builder.problem.addConstraint(
                lambda x, y: not p(x, y),
                [(0, "position"), (i, "position")]
            )

        scene = scene_builder.build()
        
        return scene

class Question6(Question):
    def __init__(self, config, relation):
        self.config = config
        self.answer_type = AnswerType.YES_OR_NO
        self.relation = relation #random.choice([Relation.SHAPE, Relation.COLOUR])
        if self.relation == Relation.SHAPE:
            self.primary_aspect = Aspect.COLOUR
            self.primary_colour = Colour.random()
            self.primary_shape = None
            self.secondary_aspect = Aspect.COLOUR
            self.secondary_colour = Colour.random(excluding=self.primary_colour)
            self.secondary_shape = None
        else:
            self.primary_aspect = Aspect.SHAPE
            self.primary_colour = None
            self.primary_shape = Shape.random()        
            self.secondary_aspect = Aspect.SHAPE
            self.secondary_colour = None
            self.secondary_shape = Shape.random(excluding=self.primary_shape)


    def __str__(self):
        question = "Are the {} and the {} of the same {}?".format(
            self.primary_label(),
            self.secondary_label(),
            "shape" if self.relation == Relation.SHAPE else "colour"
        )
        return question

    def get_random_answer(self):
        return random.choice([True, False])

    def get_all_possible_answers(self):
        return [True, False]

    def get_random_scene(self, answer):
        scene_builder = SceneBuilder(self.config)

        scene_builder.set_aspect(0, self.primary_aspect, self.primary_shape, self.primary_colour, unique=True)
        scene_builder.set_aspect(1, self.secondary_aspect, self.secondary_shape, self.secondary_colour, unique=True)

        scene_builder.problem.addConstraint(
            lambda x, y: x == y if answer else x != y,
            [(i,  "shape" if self.relation == Relation.SHAPE else "colour") for i in range(2)]
        )

        scene = scene_builder.build()
        
        return scene

class Question7(Question):
    def __init__(self, config, relation):
        self.config = config
        self.answer_type = AnswerType.YES_OR_NO
        self.relation = relation #random.choice([Relation.SHAPE, Relation.COLOUR])
        if self.relation == Relation.SHAPE:
            self.primary_aspect = Aspect.COLOUR
            self.primary_colour = Colour.random()
            self.primary_shape = None
        else:
            self.primary_aspect = Aspect.SHAPE
            self.primary_colour = None
            self.primary_shape = Shape.random()        
        self.secondary_aspect = None
        self.secondary_colour = None
        self.secondary_shape = None


    def __str__(self):
        question = "Are all {}s of the same {}?".format(
            self.primary_label(),
            "shape" if self.relation == Relation.SHAPE else "colour"
        )
        return question

    def get_random_answer(self):
        return random.choice([True, False])

    def get_all_possible_answers(self):
        return [True, False]
        
    def get_random_scene(self, answer):
        scene_builder = SceneBuilder(self.config)

        how_many = random.choice([1, 2, 3] if answer else [2, 3])

        for i in range(0, how_many):
            scene_builder.set_aspect(i, self.primary_aspect, self.primary_shape, self.primary_colour)
        
        for i in range(how_many, 5):
            scene_builder.problem.addConstraint(
                lambda x: x != (self.primary_shape if self.primary_aspect == Aspect.SHAPE else self.primary_colour),
                [(i, "shape" if self.primary_aspect == Aspect.SHAPE else "colour")]
            )
        
        scene_builder.problem.addConstraint(
            lambda x, *ys: all([x == y for y in ys]) if answer else not all([x == y for y in ys]),
            [(i, "shape" if self.relation == Relation.SHAPE else "colour") for i in range(how_many)]
        )

        scene = scene_builder.build()
        
        return scene

def answer_to_text(config, answer):
    if isinstance(answer, Colour):
        return config.concrete_colours[answer].name
    elif isinstance(answer, Shape):
        return config.concrete_shapes[answer].name
    elif isinstance(answer, Relation):
        return {
            Relation.FURTHEST: "furthest",
            Relation.CLOSEST: "closest",
            Relation.LEFT: "left",
            Relation.RIGHT: "right",
            Relation.ABOVE: "above",
            Relation.BELOW: "below"
        }[answer]
    elif isinstance(answer, bool):
        return "yes" if answer else "no"
    else:
        return answer


def dumb_encode_answer(answer):
    answer_vector = torch.zeros(3)
    if isinstance(answer, Colour):
        answer_vector[answer.value] = 1
    elif isinstance(answer, Shape):
        answer_vector[answer.value] = 1
    else:
        raise ValueError('cant encode this shit')
    return answer_vector

def dumb_answer_label(answer):
    if isinstance(answer, Colour):
        return answer.value
    elif isinstance(answer, Shape):
        return answer.value
    elif isinstance(answer, Relation):
        return {
            Relation.LEFT: 0,
            Relation.RIGHT: 1,
            Relation.ABOVE: 0,
            Relation.BELOW: 1,
            Relation.FURTHEST: 0,
            Relation.CLOSEST: 1
        }[answer]
    elif isinstance(answer, bool):
        return int(answer)
    else:
        return answer - 1
        #raise ValueError('cant encode this shit')


QUESTION_MAPPING = {
    "What is the colour of the object furthest away from the <COLOUR> object?": # -
        (Question1, (AnswerType.COLOUR, Relation.FURTHEST, Aspect.COLOUR)),
    "What is the colour of the object furthest away from the <SHAPE>?": 
        (Question1, (AnswerType.COLOUR, Relation.FURTHEST, Aspect.SHAPE)),
    "What is the colour of the object closest to from the <COLOUR> object?":
        (Question1, (AnswerType.COLOUR, Relation.CLOSEST, Aspect.COLOUR)),
    "What is the colour of the object closest to from the <SHAPE>?": # -
        (Question1, (AnswerType.COLOUR, Relation.CLOSEST, Aspect.SHAPE)),
    "What is the shape of the object furthest away from the <COLOUR> object?": # -
        (Question1, (AnswerType.SHAPE, Relation.FURTHEST, Aspect.COLOUR)),
    "What is the shape of the object furthest away from the <SHAPE>?":
        (Question1, (AnswerType.SHAPE, Relation.FURTHEST, Aspect.SHAPE)),
    "What is the shape of the object closest to from the <COLOUR> object?":
        (Question1, (AnswerType.SHAPE, Relation.CLOSEST, Aspect.COLOUR)),
    "What is the shape of the object closest to from the <SHAPE>?": # -
        (Question1, (AnswerType.SHAPE, Relation.CLOSEST, Aspect.SHAPE)),

    "What is the colour of the <SHAPE> on the left of the <COLOUR> object?": # -
        (Question2, (AnswerType.COLOUR, Relation.LEFT)),
    "What is the colour of the <SHAPE> on the right of the <COLOUR> object?":
        (Question2, (AnswerType.COLOUR, Relation.RIGHT)),
    "What is the colour of the <SHAPE> above the <COLOUR> object?": # - 
        (Question2, (AnswerType.COLOUR, Relation.ABOVE)),
    "What is the colour of the <SHAPE> below the <COLOUR> object?":
        (Question2, (AnswerType.COLOUR, Relation.BELOW)),
    "What is the shape of the <COLOUR> on the left of the <SHAPE> object?":
        (Question2, (AnswerType.SHAPE, Relation.LEFT)),
    "What is the shape of the <COLOUR> on the right of the <SHAPE> object?": # -
        (Question2, (AnswerType.SHAPE, Relation.RIGHT)),
    "What is the shape of the <COLOUR> above the <SHAPE> object?":
        (Question2, (AnswerType.SHAPE, Relation.ABOVE)),
    "What is the shape of the <COLOUR> below the <SHAPE> object?": # -
        (Question2, (AnswerType.SHAPE, Relation.BELOW)),

    "Is the <SHAPE> on the left or on the right of the <SHAPE>?": # -
        (Question3, (AnswerType.LEFT_OR_RIGHT, Aspect.SHAPE, Aspect.SHAPE)),
    "Is the <COLOUR> object on the left or on the right of the <SHAPE>?":
        (Question3, (AnswerType.LEFT_OR_RIGHT, Aspect.SHAPE, Aspect.COLOUR)),
    "Is the <SHAPE> on the left or on the right of the <COLOUR> object?": # -
        (Question3, (AnswerType.LEFT_OR_RIGHT, Aspect.COLOUR, Aspect.SHAPE)),
    "Is the <COLOUR> object on the left or on the right of the <COLOUR> object?":
        (Question3, (AnswerType.LEFT_OR_RIGHT, Aspect.COLOUR, Aspect.COLOUR)),
    "Is the <SHAPE> above or below the <SHAPE>?":
        (Question3, (AnswerType.ABOVE_OR_BELOW, Aspect.SHAPE, Aspect.SHAPE)), # -
    "Is the <COLOUR> object above or below the <SHAPE>?":
        (Question3, (AnswerType.ABOVE_OR_BELOW, Aspect.SHAPE, Aspect.COLOUR)),
    "Is the <SHAPE> above or below the <COLOUR> object?":
        (Question3, (AnswerType.ABOVE_OR_BELOW, Aspect.COLOUR, Aspect.SHAPE)), #-
    "Is the <COLOUR> object above or below the <COLOUR> object?":
        (Question3, (AnswerType.ABOVE_OR_BELOW, Aspect.COLOUR, Aspect.COLOUR)),
    "Is the <SHAPE> closest or furthest from the <SHAPE>?":
        (Question3, (AnswerType.CLOSEST_OR_FURTHEST, Aspect.SHAPE, Aspect.SHAPE)), # -
    "Is the <COLOUR> object closest or furthest from the <SHAPE>?":
        (Question3, (AnswerType.CLOSEST_OR_FURTHEST, Aspect.SHAPE, Aspect.COLOUR)),
    "Is the <SHAPE> closest or furthest from the <COLOUR> object?":
        (Question3, (AnswerType.CLOSEST_OR_FURTHEST, Aspect.COLOUR, Aspect.SHAPE)), # -
    "Is the <COLOUR> object closest or furthest from the <COLOUR> object?":
        (Question3, (AnswerType.CLOSEST_OR_FURTHEST, Aspect.COLOUR, Aspect.COLOUR)),

    "How many objects are of the same shape as the <COLOUR> object?":
        (Question4, (Relation.COLOUR,)),
    "How many objects are of the same colour as the <SHAPE>?":
        (Question4, (Relation.SHAPE,)),

    "How many objects are on the left of the <SHAPE>?": # -
        (Question5, (Relation.LEFT, Aspect.SHAPE)),
    "How many objects are on the left of the <COLOUR> object?":
        (Question5, (Relation.LEFT, Aspect.COLOUR)),
    "How many objects are on the right of the <SHAPE>?":
        (Question5, (Relation.RIGHT, Aspect.SHAPE)),
    "How many objects are on the right of the <COLOUR> object?": # -
        (Question5, (Relation.RIGHT, Aspect.COLOUR)),
    "How many objects are above the <SHAPE>?": # -
        (Question5, (Relation.ABOVE, Aspect.SHAPE)),
    "How many objects are above the <COLOUR> object?":
        (Question5, (Relation.ABOVE, Aspect.COLOUR)),
    "How many objects are below the <SHAPE>?":
        (Question5, (Relation.BELOW, Aspect.SHAPE)),
    "How many objects are below the <COLOUR> object?": # -
        (Question5, (Relation.BELOW, Aspect.COLOUR)),

    "Are the <COLOUR> and the <COLOUR> objects of the same shape?":
        (Question6, (Relation.COLOUR,)),
    "Are the <SHAPE> and the <SHAPE> of the same colour?":
        (Question6, (Relation.SHAPE,)),

    "Are all <COLOUR> objects of the same shape?":
        (Question7, (Relation.SHAPE,)),
    "Are all <SHAPE> of the same colour?":
        (Question7, (Relation.COLOUR,))
}


class AlmostClevrDataset(Dataset):
    def __init__(self):
        self.config = config = Config(
            concrete_shapes={
                Shape.SHAPE_0: Square,
                Shape.SHAPE_1: Circle,
                Shape.SHAPE_2: Triangle
            },
            concrete_colours={
                Colour.COLOUR_0: ConcreteColour("red", 255, 0, 0),
                Colour.COLOUR_1: ConcreteColour("green", 0, 255, 0),
                Colour.COLOUR_2: ConcreteColour("blue", 0, 0, 255)
            },
            grid_size=48, #64
            grid_step=8, #17
            shape_size=6 #6
        )

        self.question_types = [
            Question1,
            Question2,
            Question3,
            Question4,
            Question5,
            Question6,
            Question7
        ]

    def __getitem__(self, index):
        random.seed(index)
        question_template = random.choice(list(QUESTION_MAPPING.keys()))
        question_class, question_args = QUESTION_MAPPING[question_template]
        print(question_class, question_args)
        question = question_class(self.config, *question_args)
        answer = question.get_random_answer()
        scene = question.get_random_scene(answer)
        print(question, answer)
        return (question.to_tensor(), scene.to_tensor(), dumb_answer_label(answer))


    def __len__(self):
        return 2


class AlmostClevrFewShotDataset(Dataset):
    def __init__(self, question_templates, size=10000, n=5, target_size=100, seed=0):
        self.size = size
        self.n = n
        self.target_size = target_size
        self.seed = seed
        self.config = Config(
            concrete_shapes={
                Shape.SHAPE_0: Square,
                Shape.SHAPE_1: Circle,
                Shape.SHAPE_2: Triangle
            },
            concrete_colours={
                Colour.COLOUR_0: ConcreteColour("red", 255, 0, 0),
                Colour.COLOUR_1: ConcreteColour("green", 0, 255, 0),
                Colour.COLOUR_2: ConcreteColour("blue", 0, 0, 255)
            },
            grid_size=48,
            grid_step=8,
            shape_size=6
        )
        self.question_templates = question_templates

    def __getitem__(self, index):
        random.seed(self.seed + index)

        question_template = random.choice(self.question_templates)
        question_class, question_args = QUESTION_MAPPING[question_template]
        question = question_class(self.config, *question_args)
        answers = question.get_all_possible_answers()
        k = len(answers)

        support_set = torch.empty(k, self.n, self.config.grid_size, self.config.grid_size, 3)
        target_set = torch.empty(self.target_size, self.config.grid_size, self.config.grid_size, 3)
        support_labels = torch.empty(k, self.n)
        target_labels = torch.empty(self.target_size)

        for i, answer in enumerate(question.get_all_possible_answers()):
            for j in range(self.n):
                scene = question.get_random_scene(answer)
                support_set[i][j] = scene.to_tensor()
                support_labels[i][j] = dumb_answer_label(answer)
        
        for i in range(self.target_size):
            answer = question.get_random_answer()
            scene = question.get_random_scene(answer)
            target_set[i] = scene.to_tensor()
            target_labels[i] = dumb_answer_label(answer)
        
        return (question.to_tensor(), support_set, target_set, support_labels, target_labels)

    def __len__(self):
        return self.size


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = AlmostClevrFewShotDataset(question_templates=[
        "What is the colour of the object closest to from the <SHAPE>?",
        "How many objects are of the same shape as the <COLOUR> object?"
    ], n=1)
    question, support_set, target_set, support_labels, target_labels = dataset[1]
    for i in range(0, support_set.shape[0]):
        for j in range(0, support_set.shape[1]):
            plt.subplot(support_set.shape[0], support_set.shape[1], i * support_set.shape[1] + j + 1)
            plt.imshow(support_set[i][j].long())
    plt.show()

# Question type 1:
# -----------
# What is the colour of the object furthest away from the <COLOUR> object?
# What is the colour of the object furthest away from the <SHAPE>?
# What is the colour of the object closest to from the <COLOUR> object?
# What is the colour of the object closest to from the <SHAPE>?
# What is the shape of the object furthest away from the <COLOUR> object?
# What is the shape of the object furthest away from the <SHAPE>?
# What is the shape of the object closest to from the <COLOUR> object?
# What is the shape of the object closest to from the <SHAPE>?

# Question type 2:
# -----------
# What is the colour of the <SHAPE> on the left of the <COLOUR> object?
# What is the colour of the <SHAPE> on the right of the <COLOUR> object?
# What is the colour of the <SHAPE> above the <COLOUR> object?
# What is the colour of the <SHAPE> below the <COLOUR> object?
# What is the shape of the <COLOUR> on the left of the <COLOUR> object?
# What is the shape of the <COLOUR> on the right of the <COLOUR> object?
# What is the shape of the <COLOUR> above the <COLOUR> object?
# What is the shape of the <COLOUR> below the <COLOUR> object?

# Question type 3:
# -----------
# Is the <SHAPE> on the left or on the right of the <SHAPE>?
# Is the <COLOUR> object on the left or on the right of the <SHAPE>?
# Is the <SHAPE> on the left or on the right of the <COLOUR> object?
# Is the <COLOUR> object on the left or on the right of the <COLOUR> object?
# Is the <SHAPE> above or below the <SHAPE>?
# Is the <COLOUR> object above or below the <SHAPE>?
# Is the <SHAPE> above or below the <COLOUR> object?
# Is the <COLOUR> object above or below the <COLOUR> object?
# Is the <SHAPE> closest or furthest from the <SHAPE>?
# Is the <COLOUR> object closest or furthest from the <SHAPE>?
# Is the <SHAPE> closest or furthest from the <COLOUR> object?
# Is the <COLOUR> object closest or furthest from the <COLOUR> object?

# Question type 4:
# -----------
# How many objects are of the same shape as the <COLOUR> object?
# How many objects are of the same colour as the <SHAPE>?

# Question type 5:
# -----------
# How many objects are on the left of the <SHAPE>?
# How many objects are on the left of the <COLOUR> object?
# How many objects are on the right of the <SHAPE>?
# How many objects are on the right of the <COLOUR> object?
# How many objects are above the <SHAPE>?
# How many objects are above the <COLOUR> object?
# How many objects are below the <SHAPE>?
# How many objects are below the <COLOUR> object?

# Question type 6:
# -----------
# Are the <COLOUR> and the <COLOUR> objects of the same shape?
# Are the <SHAPE> and the <SHAPE> of the same colour?

# Question type 7:
# -----------
# Are all <COLOUR> objects of the same shape?
# Are all <SHAPE> of the same colour?