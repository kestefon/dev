# -*- coding: utf-8 -*-

# Wiley Boilerplate Dash App
# Wiley Intelligent Solutions 2018

import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import os
import search_engine
import time
from rq import Queue  # requires Redis server (see readme)
from worker import conn  # worker.py handles the connection to Redis
import uuid

# initialize app
app = dash.Dash(__name__, static_folder='static')  # config to enable
app.scripts.config.serve_locally = True  # things like css to
app.css.config.serve_locally = True  # be served locally from /static
server = app.server  # folder

# get static images (recommended method is to load images as base64 strings)
static_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static')
wiley_logo = base64.b64encode(open(os.path.join(static_folder, 'wiley.png'), 'rb').read())
robot_logo = base64.b64encode(open(os.path.join(static_folder, 'robot.png'), 'rb').read())

# markdown docs for layout
divider_markdown = '''
***
'''
description_markdown = '''
This is our boilerplate Dash app.
 '''
query_help_markdown = '''
We often use processes that take a while that would otherwise cause server and browser timeouts.
This app uses a background worker, automatic refreshing and a spinner to prevent timeouts and
provide an improved user experience.
'''

# initialize app layout
app.layout = html.Div(children=[

    # load stylesheets locally since they don't fetch from remote
    # locations when app is running on Heroku
    html.Link(href='/static/codepen.css', rel='stylesheet'),
    html.Link(href='/static/load_screen.css', rel='stylesheet'),

    # Our team's logo
    html.Div(
        html.Img(id='robot-logo',
                 src='data:image/png;base64,{}'.format(robot_logo.decode()),
                 style={'width': '100px'}), style={'display': 'inline', 'float': 'right', 'vertical-align': 'middle'}),

    # app name and description
    html.H1(children='Wiley Boilerplate Dash App'),
    dcc.Markdown(description_markdown),
    html.Br(),

    # query form and submit button
    # help text
    dcc.Markdown(query_help_markdown),
    html.Br(),

    # Submit
    html.Label('Press submit to start a 2 second process:'),
    html.Br(),
    html.Button(id='submit', type='submit', children='Submit'),
    html.Br(),
    html.Br(),

    # status infomation, e.g. "please wait"
    html.Div(id='status'),

    # invisible div to safely store the current job-id
    html.Div(id='job-id', style={'display': 'show'}),

    # this div is the target of the refresh during querying
    # initially there is no refresh (interval=1 hour) but during
    # a query it refreshes regularly until the results are ready
    html.Div([

        html.Div(children='', id='dummy-results'),
        dcc.Interval(
            id='update-interval',
            interval=60 * 60 * 5000,  # in milliseconds
            n_intervals=0
        )

    ], id='results'),

    # footer with corporate branding 
    dcc.Markdown(children=divider_markdown),
    html.Div([
        html.Div(children='Put your legal notices, corporate logo or whatever here',
                 style={'text-align': 'left', 'display': 'inline-block', 'vertical-align': 'middle'}),
    ], style={'display': 'inline-block', 'vertical-align': 'middle'}),
    html.Div(
        html.Img(id='wiley-logo',
                 src='data:image/png;base64,{}'.format(wiley_logo.decode()),
                 style={'width': '150px'}), style={'display': 'inline', 'float': 'right', 'vertical-align': 'middle'})

], style={'padding': '10px 10px'})


# this callback checks submits the query as a new job, returning job_id to the invisible div
@app.callback(
    dash.dependencies.Output('job-id', 'children'),
    [dash.dependencies.Input('submit', 'n_clicks')])
def query_submitted(click):
    print("query submitted callback")
    if click == 0 or click is None:
        print("click: ", click)
        return ''
    else:

        print("a query was submitted, so queue it up and return job_id")
        duration = 2  # pretend the process takes 2 seconds to complete
        print("query submitted callback CONNECTION value:", conn)
        q = Queue(connection=conn)
        job_id = str(uuid.uuid4())
        print("enqueue call")
        job = q.enqueue_call(func=search_engine.run_query,
                             args=(duration, 'Blackburn, Lancs'),
                             timeout='5',
                             job_id=job_id)

        print("job_id:", job_id)
        return job_id


# this callback checks if the job result is ready.  If it's ready
# the results return to the table.  If it's not ready, it pauses
# for a short moment, then empty results are returned.  If there is
# no job, then empty results are returned. 
@app.callback(
    dash.dependencies.Output('dummy-results', 'children'),
    [dash.dependencies.Input('update-interval', 'n_intervals')],
    [dash.dependencies.State('job-id', 'children')])
def update_results_tables(n_intervals, job_id):
    print("starting update results table")
    q = Queue(connection=conn)
    print("update results table q: ", q)
    print("update results table CONNECTION: ", conn)
    queued_job_ids = q.job_ids  # Gets a list of job IDs from the queue
    queued_jobs = q.jobs  # Gets a list of enqueued job instances
    print("update results table queued_job_ids:", queued_job_ids)
    print("update results table queued jobs", queued_jobs)
    print("length of queued_job_ids list:", len(queued_job_ids))
    job = q.fetch_job(job_id)
    print("update results table job:", job)
    print("update results table job status:", job.status)
    if job is not None:
        print("update_results_table: job exists - try to get result")
        result = job.result
        print("update_results_table: result: ", result)
        if result is None:
            print('''update_results_table:
             results aren't ready, pause then return empty results
             You will need to fine tune this interval depending on
             your environment''')
            time.sleep(3)
            return ''
        if result is not None:
            print("results are ready")
            return result
    else:
        # no job exists with this id
        return ''


# this callback orders the table to be regularly refreshed if
# the user is waiting for results, or to be static (refreshed once
# per hour) if they are not.
@app.callback(
    dash.dependencies.Output('update-interval', 'interval'),
    [dash.dependencies.Input('job-id', 'children'),
     dash.dependencies.Input('update-interval', 'n_intervals')])
def stop_or_start_table_update1(job_id, n_intervals):
    print("stop or start_table_update1")
    q = Queue(connection=conn)
    print("stop or start_table_update1 CONNECTION:", conn)
    job = q.fetch_job(job_id)
    print("job:", job)
    if job is not None:
        print("the job exists, try to get results")
        # the job exists - try to get results
        result = job.result
        print("result", result)
        if result is None:
            print("result is None")
            # a job is in progress but we're waiting for results
            # therefore regular refreshing is required.  You will
            # need to fine tune this interval depending on your
            # environment.
            return 1000
        else:
            print("the results are ready, therefore stop regular refreshing")
            return 60 * 60 * 1000
    else:
        print("the job does not exist, therefore stop regular refreshing")
        return 60 * 60 * 1000


# this callback displays a please wait message in the status div if
# the user is waiting for results, or nothing if they are not.
@app.callback(
    dash.dependencies.Output('status', 'children'),
    [dash.dependencies.Input('job-id', 'children'),
     dash.dependencies.Input('update-interval', 'n_intervals')])
def stop_or_start_table_update2(job_id, n_intervals):
    print("stop or start table update2")
    print('''stop or start table update2: this callback displays a please wait message in the status div if
    the user is waiting for results, or nothing if they are not.''')
    q = Queue(connection=conn)
    job = q.fetch_job(job_id)
    print("stop or start table update2 CONNECTION:", conn)
    print("stop or start table update2 q:", q)
    print("stop or start table update2 job:", job)
    if job is not None:
        print("the job exists - try to get results")
        result = job.result
        print("result", result)
        if result is None:
            print("a job is in progress and we're waiting for results")
            return 'Running query.  This might take a moment - don\'t close your browser!'
        else:
            print("the results are ready, therefore no message")
            return ''
    else:
        print("the job does not exist, therefore no message")
        return ''


# start the app
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, processes=1)