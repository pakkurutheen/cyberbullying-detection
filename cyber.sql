-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Feb 20, 2024 at 02:07 PM
-- Server version: 10.4.25-MariaDB
-- PHP Version: 8.1.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `cyber`
--

-- --------------------------------------------------------

--
-- Table structure for table `tweets`
--

CREATE TABLE `tweets` (
  `name` varchar(32) DEFAULT NULL,
  `date` varchar(32) DEFAULT NULL,
  `tweet` varchar(32) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `tweets`
--

INSERT INTO `tweets` (`name`, `date`, `tweet`) VALUES
('sdpro', '2023-02-08 12:01:51.420457', 'hello jas'),
('sdpro', '2023-02-08 12:02:08.423885', 'bitch'),
('sdpro', '2023-03-15 20:45:22.986009', 'hi'),
('sdpro', '2023-03-15 20:45:29.875630', 'fuck ...its'),
('sdpro', '2023-03-15 22:05:46.680289', 'hi'),
('sdpro', '2023-03-15 22:06:11.857046', 'fuck '),
('sdpro', '2023-03-18 12:16:21.218551', 'hi'),
('sdpro', '2023-03-18 12:16:30.863392', 'bitch'),
('abc', '2023-03-18 12:18:14.142634', 'hello jas'),
('abc', '2023-03-18 12:18:19.462848', 'fuck ...its'),
('sdpro', '2023-03-23 16:54:59.085441', 'hi'),
('sdpro', '2023-03-23 16:55:09.727076', 'bitch'),
('abc', '2023-03-25 17:44:28.177895', 'hello'),
('abc', '2023-03-25 17:44:37.078160', 'bitch'),
('sdpro', '2024-02-14 10:45:45.466395', 'fuck'),
('sdpro', '2024-02-14 10:52:26.378700', 'fuck'),
('sdpro', '2024-02-14 10:58:02.873518', 'fuck');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `name` varchar(32) DEFAULT NULL,
  `phone` varchar(32) DEFAULT NULL,
  `password` varchar(32) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`name`, `phone`, `password`) VALUES
('sdpro', '9876543210', 'sd'),
('abc', '1234567890', '123'),
('admin', '9876541230', '1234');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
