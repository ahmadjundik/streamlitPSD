from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
import os

# Hubungkan ke workspace Anda
ws = Workspace.from_config()

# Registrasikan model Anda
model = Model.register(workspace=ws,
                       model_path="Users/khairurrijal/project-directory/modelxgb.sav",  # Path relatif ke model Anda
                       model_name="modelxgboost")

# Buat environment baru
env = Environment(name="xgboost-env")

# Tentukan dependensi
deps = CondaDependencies.create(pip_packages=["xgboost", "azureml-defaults"])

# Set Conda dependencies
env.python.conda_dependencies = deps

# Registrasikan environment
env.register(workspace=ws)

# Memeriksa direktori kerja saat ini dan keberadaan direktori src serta file score.py
current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")

src_dir = os.path.join(current_working_directory, 'src')
score_script = os.path.join(src_dir, 'score.py')

if os.path.isdir(src_dir) and os.path.isfile(score_script):
    print("Directory src and file score.py found.")
    inference_config = InferenceConfig(entry_script="score.py", source_directory="src", environment=env)
else:
    raise Exception("Directory src or file score.py not found. Please check the paths.")

# Konfigurasi penyebaran
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy model sebagai web service
service = Model.deploy(workspace=ws,
                       name="xgboost-service",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)

