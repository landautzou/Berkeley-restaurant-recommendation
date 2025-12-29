"""A Yelp-powered Restaurant Recommendation Program"""
# Partner A Full Name: landau
# Partner B Full Name: devin
# Period #: 1

# COMPLETE QUESTIONS 3-8

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################

# BEFORE CODING Q3-4, WATCH & TAKE NOTES ON:
# "PROJ 02 Phases 1 & 2 Videos: Key terms, k-means, & Q2-4" 
# (Google Classroom > Projects)

def find_closest(location, centroids):
    """Return the centroid in the list centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [4.0, 3.0], [5.0, 5.0], [2.0, 3.0]])
    [2.0, 3.0]
    # Driver Name: landau
    # Navigator Name: devin
    # Add comments explaining how your implementation works
    #location (first input) is a list representing coordinates; centroids (second input) is a list of lists, 
        where each list pair represents the list of a centroid
    for each centroid in the list of centroids
        it calculates the distance between the location and the centroid using the function distance
    each item in the new list dist_lst is a list with centroid, distance_value
        centroid is the centroid's coordinates
        distance_value is the distance between location and centroid

    min() finds the item with the smallest distance (second argument)
    the key function tells min to compare the second element (distance)
    [0] at the very end retrieves only the centroid, not the distance
        so the returned value is the centroid
    """
    # BEGIN Question 3
    dist_lst=[[centroid, distance(location, centroid)] for centroid in centroids]
    return min(dist_lst, key=lambda x: x[1])[0]
    # END Question 3
print(find_closest([3.0, 4.0], [[0.0, 0.0], [4.0, 3.0], [5.0, 5.0], [2.0, 3.0]])) # [2.0, 3.0]
print()

def group_by_first(pairs):
    """Return a list of lists that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)  # Values from pairs that start with 1, 3, and 2 respectively
    [[2, 3, 2], [2, 1], [4]]
    """
    # GIVEN - USE IN GROUP_BY_CENTROID BELOW
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]
example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
print(group_by_first(example))  # Values from pairs that start with 1, 3, and 2 respectively: [[2, 3, 2], [2, 1], [4]]

def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
   
    resturants is a list of restaurant locations
    centroids is a list of centroid locations
    """
    # Driver Name: Devin
    # Navigator Name: Landau
    # Add comments explaining how your implementation works
    '''
    you first want to create a list of pairs for each restaurant
    find_closest(restaurant, centoids) determines which centroid is closest to the current restaurant
        it returns the closest centroid's coordinates
    it results in a list where each element(list) in the list is a pair of the closest centroid and the restaurant itself

    because it returns a list of lists (closest centroid to the restaurant, restaurants coordinates)
        group_by_first groups all the restaurants by their first element, the closest centroid
    it returns a list of clusters whree each cluster contains all restaurants that are closest to that centroid
    '''

    # BEGIN Question 4
    pairs=[[find_closest(restaurant, centroids), restaurant] for restaurant in restaurants]
    return group_by_first(pairs)
    # END Question 4


# BEFORE CODING Q5-6, WATCH & TAKE NOTES ON:
# "PROJ 02 Phase 2 Videos: Hints for Q5-6" 
# (Google Classroom > Projects)

def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    # Driver Name: landau
    # Navigator Name: devin
    # Add comments explaining how your implementation works

    # BEGIN Question 5
    #the centroid is a point that is close to multiple restaurants
    #by finding the latitude of the centroid, you want to find the average of all latitudes of restaurnats
    #to find the longitude of the centroid, you want to find the average of the all the longitudes
    #then you will return the latitude and longitude as a list pair coordinate
    centroid_latitude = mean([restaurant_location(r)[0] for r in cluster])
    centroid_longitude = mean([restaurant_location(r)[1] for r in cluster])
    return [centroid_latitude, centroid_longitude]
    # END Question 5

cluster1 = [
make_restaurant('A', [-3, -3], [], 3, [make_review('A', 2)]),
make_restaurant('B', [1, -2],  [], 1, [make_review('B', 1)]),
make_restaurant('C', [2, -2.5],  [], 1, [make_review('C', 5)]),
]

print(find_centroid(cluster1))
print()

def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0

    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
    # Driver Name: landau
    # Navigator Name: devin
    # Add comments explaining how your implementation works
    #you first group the restaurants by centroid
    #you then find the new centroid
        #this iterates until the truest centroid is found
        # BEGIN Question 6
        cluster=group_by_centroid(restaurants, centroids)
        centroids=[find_centroid(cluster) for cluster in cluster]
        # END Question 6
        n += 1
    return centroids


################################
# Phase 3: Supervised Learning #
################################

# BEFORE CODING Q7-8, WATCH & TAKE NOTES ON:
# "PROJ 02 Phase 3 Videos: Supervised Learning & Hints for Q7-8" 
# (Google Classroom > Projects)

def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    xs = [feature_fn(r) for r in restaurants]
    ys = [user_rating(user, restaurant_name(r)) for r in restaurants]

    # Driver Name:
    # Navigator Name:
    # Add comments explaining how your implementation works
    '''
    general things to know:
        mean(..) takes in 1 input, and returns the mean
        r squared is how well it fits the values; how well it explains variance

    the code is all derived from formulas in the notes slide
    s_xx: for each x value in xs, it finds the deviation from the mean(x-mean(xs)) and squares it
        sum of squares for x values
    s_yy: for each y value in ys, it finds the deviation from the mean(y-mean(ys)) and squares it
        sum of squares for x values
    s_xy: 
        zip() pairs each x with its corresponding y value
        for each paired value, it calculates the product of the devation of x from mean(xs)
            and devation of y from mean(ys)--(x-mean(xs))*(y-mean(ys))
    b is the slope of the line, how much y changes per unit x
    a is the y-intercept

    '''
    # BEGIN Question 7
    s_xx=sum([(x-mean(xs))**2 for x in xs])
    s_yy=sum([(x-mean(ys))**2 for y in ys])
    s_xy=sum([(x-mean(xs))*(y-mean(ys)) for x, y in zip(xs, ys)])
    b, a, r_squared=s_xy/s_xx, mean(ys)-b*mean(xs), (s_xy**2)/(s_xx*s_yy)
    # END Question 7

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    # Driver Name: Devin
    # Navigator Name: Landau
    # Add comments explaining how your implementation works
    #[find_predictor(user, reviewed, ff) for ff in feature_fns]: the first input for max
        #creates a list of lists, each sublist contains a predictor function for the feature function and the r^2 value
        #(predictor, r^2)
    #max(..., key=lambda z:z[1]) finds the sublist with the highest first index, the highest r^2 value
        #the key function tells max to compare the second item
    #[0] extracts the first element (the best predictor function, based on highest r^2 value)
        #this is what it returns
    # BEGIN Question 8
    return max([find_predictor(user, reviewed, fn) for fn in feature_fns], key=lambda x: x[1])[0]
    # END Question 8


# QUESTION 9 IS NOT REQUIRED. OPTIONAL CHALLENGE.
def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 9
    ratings = {}
    for restaurant in restaurants:
        name = restaurant_name(restaurant)
        if restaurant in reviewed:
            ratings[name] = user_rating(user, name)
        else:
            ratings[name] = predictor(restaurant)
    return ratings
    # END Question 9

# QUESTION 10 IS NOT REQUIRED. OPTIONAL CHALLENGE.
def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    return [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]
    # END Question 10n


def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(list(CATEGORIES), 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
