"""fakenewsdetect URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from fakenews import views
from django.urls import include, re_path

urlpatterns = [
    re_path('admin/', admin.site.urls),
    re_path(r'^home', views.home, name='home'),
	re_path(r'^nvb', views.nvb, name='nvb'),
	re_path(r'^pac', views.pac, name='pac'),
	re_path(r'^svm', views.svm, name='svm'),
	re_path(r'^accuracy', views.accuracy, name='accuracy'),
	re_path(r'^loginuser', views.loginuser, name='loginuser'),
	re_path(r'^rf', views.rf, name='rf'),
	re_path(r'^input', views.input, name='input'),
	re_path(r'^test', views.test, name='test'),
	re_path(r'^simple_upload', views.simple_upload, name='simple_upload'),
	re_path(r'^simple', views.simple, name='simple'),
	re_path(r'^fileshow', views.fileshow, name='fileshow'),
	re_path(r'^fileshow1', views.fileshow1, name='fileshow1'),
	re_path(r'^$', views.loginpage, name='loginpage'),
	re_path(r'^register', views.register, name='register'),
	re_path(r'^reg', views.reg, name='reg'),
	re_path(r'^svr', views.svr, name='svr'),
	re_path(r'^fertilizerRf', views.fertilizerRf, name='fertilizerRf'),
	re_path(r'^fertilizerform', views.fertilizerform, name='fertilizerform'),
        
	
]
