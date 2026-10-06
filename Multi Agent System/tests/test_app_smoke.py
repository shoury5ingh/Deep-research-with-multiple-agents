import importlib


def test_app_import_and_dummy_pipeline():
    app = importlib.import_module('app')
    assert hasattr(app, 'step_card')

    from pipeline import run_research_pipeline
    result = run_research_pipeline('test topic')

    assert isinstance(result, dict)
    assert 'search_results' in result
    assert 'scraped_content' in result
    assert 'report' in result
    assert 'feedback' in result
    assert 'Introduction' in result['report'] or 'Key Findings' in result['report']
