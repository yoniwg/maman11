def is_google_colaboratory_runtime():
    return get_ipython().__class__.__module__ == "google.colab._shell"

def colaboratory_enable_plotly():
  ''' for plotly working also in google colaboratory (source: https://stackoverflow.com/a/47230966/1509695) '''
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))

def enable_plotly_in_colab():
    if is_google_colaboratory_runtime():
        colaboratory_enable_plotly()
