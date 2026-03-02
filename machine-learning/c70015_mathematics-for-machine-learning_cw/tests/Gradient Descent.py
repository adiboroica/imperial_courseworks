OK_FORMAT = True

test = {   'name': 'Gradient Descent',
    'points': 4,
    'suites': [   {   'cases': [   {   'code': '>>> c = np.array([[4], [5]])\n'
                                               '>>> dummy_fn = lambda x: float(c.T @ x)\n'
                                               '>>> dummy_fn_grad = lambda x: c.T\n'
                                               '>>> dummy_start_x, dummy_start_y = 1.0, 2.0\n'
                                               '>>> \n'
                                               '>>> trajectory, minimum_loc, minimum_value = gradient_descent(dummy_fn, dummy_fn_grad, start_x=dummy_start_x, start_y=dummy_start_y, lr=0.01, '
                                               'n_steps=5, silent=True)\n'
                                               '>>> \n'
                                               '>>> target_trajectory = [np.array([[1], [2]]), np.array([[0.96], [1.95]]), np.array([[0.92], [1.9]]),\n'
                                               '...                     np.array([[0.88], [1.85]]), np.array([[0.84], [1.8]]), np.array([[0.8], [1.75]])]\n'
                                               '>>> \n'
                                               '>>> np.array([np.isclose(target.flatten(), output.flatten(), atol=1e-3) for target, output in zip(target_trajectory, trajectory)]).all()\n'
                                               'True',
                                       'failure_message': 'Gradient Descent Trajectory Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 0,
                                       'success_message': 'Gradient Descent Trajectory Test Passed'},
                                   {   'code': '>>> c = np.array([[4], [5]])\n'
                                               '>>> dummy_fn = lambda x: float(c.T @ x)\n'
                                               '>>> dummy_fn_grad = lambda x: c.T\n'
                                               '>>> dummy_start_x, dummy_start_y = 1.0, 2.0\n'
                                               '>>> \n'
                                               '>>> trajectory, minimum_loc, minimum_value = gradient_descent(dummy_fn, dummy_fn_grad, start_x=dummy_start_x, start_y=dummy_start_y, lr=0.01, '
                                               'n_steps=5, silent=True)\n'
                                               '>>> target_minimum_loc = np.array([[0.8], [1.75]])\n'
                                               '>>> np.isclose(target_minimum_loc, minimum_loc, atol=1e-3).all()\n'
                                               'True',
                                       'failure_message': 'Gradient Descent Minimum Location Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 0,
                                       'success_message': 'Gradient Descent Minimum Location Test Passed'},
                                   {   'code': '>>> c = np.array([[4], [5]])\n'
                                               '>>> dummy_fn = lambda x: float(c.T @ x)\n'
                                               '>>> dummy_fn_grad = lambda x: c.T\n'
                                               '>>> dummy_start_x, dummy_start_y = 1.0, 2.0\n'
                                               '>>> \n'
                                               '>>> trajectory, minimum_loc, minimum_value = gradient_descent(dummy_fn, dummy_fn_grad, start_x=dummy_start_x, start_y=dummy_start_y, lr=0.01, '
                                               'n_steps=5, silent=True)\n'
                                               '>>> target_minimum_value = 11.949999999999998\n'
                                               '>>> np.isclose(target_minimum_value, minimum_value, atol=1e-3)\n'
                                               'True',
                                       'failure_message': 'Gradient Descent Minimum Value Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 0,
                                       'success_message': 'Gradient Descent Minimum Value Test Passed'}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
