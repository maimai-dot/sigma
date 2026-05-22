"""Tests for SigmaConfig — dataclass defaults and overrides."""

from sigma.config import SigmaConfig


class TestSigmaConfigDefaults:
    """Default values."""

    def test_default_project_name(self):
        assert SigmaConfig().project_name == "Sigma Project"

    def test_default_creed(self):
        assert SigmaConfig().creed == ""

    def test_default_domain_keywords(self):
        assert SigmaConfig().domain_keywords == {}

    def test_default_role_map(self):
        assert SigmaConfig().role_map == {}

    def test_default_domain_agent_map(self):
        assert SigmaConfig().domain_agent_map == {}

    def test_default_action_weights(self):
        cfg = SigmaConfig()
        assert isinstance(cfg.action_weights, dict)
        assert cfg.action_weights["设计"] == 4.0
        assert cfg.action_weights["查"] == 1.0

    def test_default_constraint_keywords(self):
        cfg = SigmaConfig()
        assert isinstance(cfg.constraint_keywords, dict)
        assert cfg.constraint_keywords["必须"] == 0.5

    def test_default_lite_max_agents(self):
        assert SigmaConfig().lite_max_agents == 4

    def test_default_standard_exclude(self):
        assert SigmaConfig().standard_exclude_agents == set()

    def test_default_tool_params(self):
        assert SigmaConfig().default_tool_params == {}

    def test_default_reasonable_ranges(self):
        assert SigmaConfig().reasonable_ranges == {}

    def test_default_output_base_dir(self):
        assert SigmaConfig().output_base_dir is None

    def test_default_model_settings(self):
        cfg = SigmaConfig()
        assert cfg.default_model == "deepseek-v4-pro"
        assert cfg.default_max_tokens == 2048
        assert cfg.default_temperature == 0.2


class TestSigmaConfigOverride:
    """Field overrides."""

    def test_project_name_override(self):
        cfg = SigmaConfig(project_name="My Project")
        assert cfg.project_name == "My Project"

    def test_creed_override(self):
        cfg = SigmaConfig(creed="Be excellent")
        assert cfg.creed == "Be excellent"

    def test_domain_keywords_override(self):
        cfg = SigmaConfig(domain_keywords={"prop": ["thrust", "fuel"]})
        assert cfg.domain_keywords == {"prop": ["thrust", "fuel"]}

    def test_role_map_override(self):
        cfg = SigmaConfig(role_map={"engineer": "Lead Engineer"})
        assert cfg.role_map == {"engineer": "Lead Engineer"}

    def test_output_base_dir_override(self):
        cfg = SigmaConfig(output_base_dir="/tmp/output")
        assert cfg.output_base_dir == "/tmp/output"

    def test_reasonable_ranges_override(self):
        cfg = SigmaConfig(reasonable_ranges={"Isp": (100, 400)})
        assert cfg.reasonable_ranges == {"Isp": (100, 400)}
