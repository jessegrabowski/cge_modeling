from cge_modeling.cge import Variable, Parameter
import pytest


@pytest.mark.parametrize('cls', [Variable, Parameter], ids=['variable', 'parameter'])
@pytest.mark.parametrize('name, index, expected',
                         [
                             ('x', None, ('x', 'x, Positive = True, Real = True', ())),
                             ('x', ('i', ), ('x_{i}', 'x_{i}, Positive = True, Real = True', ('i', ))),
                             ('x', 'i', ('x_{i}', 'x_{i}, Positive = True, Real = True', ('i', ))),
                             ('x', 'i,j', ('x_{i,j}', 'x_{i,j}, Positive = True, Real = True', ('i', 'j'))),
                             ('x_d', 'i,j', ('x_{d,i,j}', 'x_{d,i,j}, Positive = True, Real = True', ('i', 'j'))),
                         ],
                         ids=['No_index', 'tuple_index', 'single_string', 'double_string', 'underscore_name'])
def test_create_variable_defaults(cls, name, index, expected):
    x = cls(name=name, index=index)
    latex_name, description, index = expected

    assert x.latex_name == latex_name
    assert x.description == description
    assert x.index == index


@pytest.mark.parametrize('cls', [Variable, Parameter], ids=['variable', 'parameter'])
def test_create_variable(cls):
    x = cls(name='x', index='i', latex_name=r'\Omega', description='A lovely variable')
    y = cls(name='y', index='i', latex_name=r'\hat{\Omega}', description='A lovely variable with a hat')

    assert x.latex_name == r'\Omega'
    assert x.description == 'A lovely variable'
    assert x.index == ('i', )
    assert y > x



