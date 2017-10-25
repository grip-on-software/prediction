pipeline {
    agent { label 'docker' }

    environment {
        GITLAB_TOKEN = credentials('prediction-gitlab-token')
    }

    parameters {
        string(name: 'PREDICTION_ARGS', defaultValue: '--label num_not_done+num_removed_stories+num_added_stories --binary --roll-sprints --roll-validation --roll-labels --replace-na --model dnn --test-interval 200 --num-epochs 1000', description: 'Prediction arguments')
    }
    options {
        gitLabConnection('gitlab')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    triggers {
        gitlab(triggerOnPush: true, triggerOnMergeRequest: true, branchFilterType: 'All', secretToken: env.GITLAB_TOKEN)
        cron('H H * * 0')
    }

    post {
        success {
            archiveArtifacts artifacts: 'output/**/*.json', excludes: 'output/sprint_labels.json', onlyIfSuccessful: true
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'success'
        }
        failure {
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'failed'
        }
        aborted {
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'canceled'
        }
    }

    stages {
        stage('Collect') {
            agent {
                docker {
                    image '$DOCKER_REGISTRY/gros-data-analysis-dashboard'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh '/bin/bash -c "rm -rf $PWD/output && mkdir $PWD/output && cd /home/docker && Rscript features.r --core --log INFO --config $ANALYSIS_CONFIGURATION --output $PWD/output"'
                }
            }
        }
        stage('Predict') {
            agent {
                docker {
                    image 'tensorflow/tensorflow:1.3.0'
                    reuseNode true
                }
            }
            steps {
                sh 'pip install --prefix /tmp/local virtualenv'
                sh 'PYTHONPATH=/tmp/local/lib/python2.7/site-packages /tmp/local/bin/virtualenv --system-site-packages /tmp/venv'
                sh ". /tmp/venv/bin/activate; pip install -r requirements.txt; python tensor.py --filename output/sprint_features.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.json"
            }
        }
        stage('Format') {
            agent {
                docker {
                    image '$DOCKER_REGISTRY/gros-data-analysis-dashboard'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh '/bin/bash -c "cd /home/docker && Rscript sprint_results.r --file $PWD/output/sprint_labels.json --config $ANALYSIS_CONFIGURATION --output $PWD/output"'
                }
            }
        }
    }
}